"""Offline HCPCS extracts from existing full hospital JSONL archives.

Reads data/{STATE}/{CCN}.jsonl (no download), keeps matching CPT/HCPCS rows,
and writes data/outputs/{CODES}/{STATE}/{CCN}.jsonl.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from .config import ScraperConfig, get_full_archive_path, get_output_path
from .models import HospitalConfig, ScrapeResult
from .utils.logger import get_logger

logger = get_logger(__name__)


def filter_jsonl_by_hcpcs(
    source: Path,
    dest: Path,
    codes: set[str],
    *,
    dry_run: bool = False,
) -> int:
    """Filter source JSONL to rows whose cpt is in codes.

    Writes dest when not dry_run. Removes an existing dest if zero rows match
    so stale extracts are not left behind.

    Returns:
        Number of matching rows
    """
    wanted = {c.upper() for c in codes}
    kept_lines: list[str] = []

    with source.open(encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if not isinstance(rec, dict):
                continue
            cpt = str(rec.get("cpt") or "").strip().upper()
            if cpt not in wanted:
                continue
            # Normalize cpt casing in output
            rec["cpt"] = cpt
            kept_lines.append(json.dumps(rec, separators=(",", ":"), ensure_ascii=False))

    if dry_run:
        return len(kept_lines)

    dest.parent.mkdir(parents=True, exist_ok=True)
    if kept_lines:
        dest.write_text("\n".join(kept_lines) + "\n", encoding="utf-8")
    elif dest.exists():
        dest.unlink()

    return len(kept_lines)


def process_hospital_from_existing(
    hospital: HospitalConfig,
    config: ScraperConfig,
    *,
    dry_run: bool = False,
) -> tuple[HospitalConfig, ScrapeResult, str]:
    """Filter one hospital's full archive into the --hcpcs output path."""
    start = time.perf_counter()
    codes = config.hcpcs_codes
    if not codes:
        raise ValueError("--from-existing requires hcpcs_codes on ScraperConfig")

    source = get_full_archive_path(config, hospital)
    dest = get_output_path(config, hospital)

    if not source.exists() or source.stat().st_size == 0:
        result = ScrapeResult.skipped(
            hospital_npi=hospital.hospital_npi,
            file_url=hospital.file_url,
            reason=f"No existing archive at {source}",
            ccn=hospital.ccn,
        )
        return hospital, result, f"- Skipped: no archive {source.name}"

    try:
        n = filter_jsonl_by_hcpcs(source, dest, set(codes), dry_run=dry_run)
    except Exception as e:
        duration = time.perf_counter() - start
        result = ScrapeResult.failure(
            hospital_npi=hospital.hospital_npi,
            file_url=str(source),
            error=e,
            duration_seconds=duration,
            ccn=hospital.ccn,
        )
        return hospital, result, f"x Failed: {e}"

    duration = time.perf_counter() - start
    result = ScrapeResult.success(
        hospital_npi=hospital.hospital_npi,
        file_url=str(source),
        records_scraped=n,
        duration_seconds=duration,
        ccn=hospital.ccn,
    )
    action = "Dry run" if dry_run else "Filtered"
    return hospital, result, f"+ {action}: {n} records from {source.name}"


def process_state_from_existing(
    hospitals: list[HospitalConfig],
    config: ScraperConfig,
    dry_run: bool = False,
) -> tuple[str, list[tuple[HospitalConfig, ScrapeResult, str]]]:
    """Filter all hospitals for one state (used as a parallel worker unit)."""
    if not hospitals:
        return "", []
    state = hospitals[0].state.upper()
    results = [
        process_hospital_from_existing(hospital, config, dry_run=dry_run)
        for hospital in hospitals
    ]
    return state, results


def run_from_existing(
    config: ScraperConfig,
    hospitals: list[HospitalConfig],
    *,
    dry_run: bool = False,
    parallel: int = 1,
) -> list[tuple[HospitalConfig, ScrapeResult, str]]:
    """Filter hospitals from existing archives.

    When parallel > 1, runs one worker per state (up to ``parallel`` states
    at a time).
    """
    if not config.hcpcs_codes:
        raise ValueError("--from-existing requires at least one --hcpcs code")

    by_state: dict[str, list[HospitalConfig]] = defaultdict(list)
    for hospital in hospitals:
        by_state[hospital.state.upper()].append(hospital)

    states = sorted(by_state.keys())
    workers = max(1, min(parallel, len(states))) if states else 1

    logger.info(
        "from_existing_started",
        hospitals=len(hospitals),
        states=len(states),
        workers=workers,
        hcpcs=config.hcpcs_codes,
        dry_run=dry_run,
    )

    if workers <= 1 or len(states) <= 1:
        return [
            process_hospital_from_existing(hospital, config, dry_run=dry_run)
            for hospital in hospitals
        ]

    ordered: list[tuple[HospitalConfig, ScrapeResult, str]] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                process_state_from_existing,
                by_state[state],
                config,
                dry_run,
            ): state
            for state in states
        }
        for future in as_completed(futures):
            state = futures[future]
            try:
                _, state_results = future.result()
            except Exception as e:
                logger.exception("from_existing_state_failed", state=state, error=str(e))
                # Surface as failures for each hospital in that state
                for hospital in by_state[state]:
                    result = ScrapeResult.failure(
                        hospital_npi=hospital.hospital_npi,
                        file_url=hospital.file_url,
                        error=e,
                        duration_seconds=0.0,
                        ccn=hospital.ccn,
                    )
                    ordered.append((hospital, result, f"x Failed: {e}"))
                continue
            ordered.extend(state_results)

    return ordered
