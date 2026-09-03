"""Price distribution stats for scraped CPT/HCPCS codes.

After a scrape, summarize each product/intervention as one row per
(code, state, price type) with: drug name (if known), HCPCS code, state,
min, 25th, median, mean, 75th, and max.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ScraperConfig, get_output_path, get_price_stats_path
from .models import HospitalConfig, ScrapeResult, ScrapeStatus

STAT_COLUMNS = [
    "drug_name",
    "hcpcs_code",
    "state",
    "type",
    "n",
    "n_hospitals",
    "min",
    "p25",
    "median",
    "mean",
    "p75",
    "max",
]

_TYPE_ORDER = {"cash": 0, "gross": 1, "net": 2}
_MONEY_COLUMNS = ("min", "p25", "median", "mean", "p75", "max")
_PREVIEW_ROWS = 25
_VALID_STATES = {
    "AL",
    "AK",
    "AZ",
    "AR",
    "CA",
    "CO",
    "CT",
    "DE",
    "FL",
    "GA",
    "HI",
    "ID",
    "IL",
    "IN",
    "IA",
    "KS",
    "KY",
    "LA",
    "ME",
    "MD",
    "MA",
    "MI",
    "MN",
    "MS",
    "MO",
    "MT",
    "NE",
    "NV",
    "NH",
    "NJ",
    "NM",
    "NY",
    "NC",
    "ND",
    "OH",
    "OK",
    "OR",
    "PA",
    "RI",
    "SC",
    "SD",
    "TN",
    "TX",
    "UT",
    "VT",
    "VA",
    "WA",
    "WV",
    "WI",
    "WY",
    "DC",
    "PR",
    "VI",
    "GU",
    "AS",
    "MP",
}


def load_concept_names(concept_path: Path) -> dict[str, str]:
    """Map CPT/HCPCS concept_code -> concept_name from Athena CONCEPT.csv.gz."""
    if not concept_path.exists():
        return {}

    df = pd.read_csv(
        concept_path,
        compression="gzip",
        sep="\t",
        usecols=["concept_code", "concept_name", "vocabulary_id"],
        dtype={"concept_code": str, "concept_name": str, "vocabulary_id": str},
        low_memory=False,
    )
    df = df[df["vocabulary_id"].isin(["CPT4", "HCPCS"])]
    names: dict[str, str] = {}
    for code, name in zip(df["concept_code"], df["concept_name"], strict=False):
        key = str(code).strip().upper()
        label = str(name).strip()
        if not key or not label or label.lower() == "nan":
            continue
        # First win; codes are unique across CPT4/HCPCS for our uses
        names.setdefault(key, label)
    return names


def collect_price_file_paths(
    config: ScraperConfig,
    results: list[tuple[HospitalConfig, ScrapeResult]],
) -> list[tuple[Path, str]]:
    """JSONL paths + state for successful or skipped hospitals with data on disk."""
    paths: list[tuple[Path, str]] = []
    seen: set[Path] = set()
    for hospital, result in results:
        if result.status not in (ScrapeStatus.SUCCESS, ScrapeStatus.SKIPPED):
            continue
        path = get_output_path(config, hospital)
        if path in seen or not path.exists() or path.stat().st_size == 0:
            continue
        seen.add(path)
        paths.append((path, hospital.state.upper()))
    return paths


def compute_price_stats(
    sources: list[Path] | list[tuple[Path, str]],
    *,
    concept_names: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Compute min/p25/median/mean/p75/max for every code, state, and price type.

    Each JSONL row is one observation. Cash, gross, and net are not mixed.
    Non-positive prices are dropped.
    """
    names = concept_names or {}
    normalized = _normalize_sources(sources)

    prices: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    hospitals: dict[tuple[str, str, str], set[str]] = defaultdict(set)

    for path, state in normalized:
        ccn = path.stem.upper()
        try:
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    rec = _parse_price_line(line)
                    if rec is None:
                        continue
                    cpt, price_type, price = rec
                    key = (cpt, state, price_type)
                    prices[key].append(price)
                    hospitals[key].add(ccn)
        except OSError:
            continue

    if not prices:
        return pd.DataFrame(columns=STAT_COLUMNS)

    rows: list[dict[str, object]] = []
    for key, values in prices.items():
        arr = np.asarray(values, dtype=np.float64)
        cpt, state, price_type = key
        rows.append(
            {
                "drug_name": names.get(cpt, ""),
                "hcpcs_code": cpt,
                "state": state,
                "type": price_type,
                "n": int(arr.size),
                "n_hospitals": len(hospitals[key]),
                "min": float(arr.min()),
                "p25": float(np.percentile(arr, 25)),
                "median": float(np.median(arr)),
                "mean": float(arr.mean()),
                "p75": float(np.percentile(arr, 75)),
                "max": float(arr.max()),
            }
        )

    df = pd.DataFrame(rows, columns=STAT_COLUMNS)
    df["_ord"] = df["type"].map(lambda t: _TYPE_ORDER.get(str(t), 9))
    df = df.sort_values(
        ["hcpcs_code", "state", "_ord"], kind="mergesort"
    ).drop(columns=["_ord"])
    for col in _MONEY_COLUMNS:
        df[col] = df[col].round(2)
    return df.reset_index(drop=True)


def write_price_stats_csv(df: pd.DataFrame, path: Path) -> Path:
    """Write stats CSV and return the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, columns=STAT_COLUMNS)
    return path


def format_stats_preview(df: pd.DataFrame, max_rows: int = _PREVIEW_ROWS) -> str:
    """Human-readable preview of price stats."""
    if df.empty:
        return "No price observations to summarize."

    shown = df.head(max_rows)
    lines = [
        "Price stats by HCPCS / state (min-p25, median-mean, p75-max):",
        f"{'code':<8} {'state':<5} {'type':<6} {'n':>7} "
        f"{'min':>9} {'p25':>9} {'median':>9} {'mean':>9} {'p75':>9} {'max':>9}  drug_name",
    ]
    for rec in shown.to_dict(orient="records"):
        drug = str(rec.get("drug_name") or "")
        if len(drug) > 40:
            drug = drug[:37] + "..."
        lines.append(
            f"{rec['hcpcs_code']:<8} {rec['state']:<5} {rec['type']:<6} {rec['n']:>7} "
            f"{rec['min']:>9.2f} {rec['p25']:>9.2f} {rec['median']:>9.2f} "
            f"{rec['mean']:>9.2f} {rec['p75']:>9.2f} {rec['max']:>9.2f}  {drug}"
        )
    remaining = len(df) - len(shown)
    if remaining > 0:
        lines.append(f"... {remaining} more rows")
    return "\n".join(lines)


def write_price_stats_for_run(
    config: ScraperConfig,
    results: list[tuple[HospitalConfig, ScrapeResult]],
    *,
    state_filter: str | None = None,
    ccn_filter: str | None = None,
) -> tuple[Path, pd.DataFrame]:
    """Compute and write one consolidated price-stats CSV for this run."""
    path = get_price_stats_path(config, state_filter=state_filter, ccn_filter=ccn_filter)
    sources = collect_price_file_paths(config, results)
    concept_names = load_concept_names(config.concept_csv_path)
    df = compute_price_stats(sources, concept_names=concept_names)
    write_price_stats_csv(df, path)
    return path, df


def _normalize_sources(
    sources: list[Path] | list[tuple[Path, str]],
) -> list[tuple[Path, str]]:
    if not sources:
        return []
    first = sources[0]
    if isinstance(first, Path):
        out: list[tuple[Path, str]] = []
        for item in sources:
            assert isinstance(item, Path)
            parent = item.parent.name.upper()
            state = parent if parent in _VALID_STATES else ""
            out.append((item, state))
        return out

    return [(Path(path), str(state).upper()) for path, state in sources]  # type: ignore[misc]


def _parse_price_line(line: str) -> tuple[str, str, float] | None:
    raw = line.strip()
    if not raw:
        return None
    try:
        rec = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(rec, dict):
        return None

    cpt = str(rec.get("cpt") or "").strip().upper()
    price_type = str(rec.get("type") or "").strip().lower()
    if not cpt or price_type not in _TYPE_ORDER:
        return None

    raw_price = rec.get("price")
    if raw_price is None:
        return None
    try:
        price = float(raw_price)
    except (TypeError, ValueError):
        return None
    if price <= 0:
        return None
    return cpt, price_type, price
