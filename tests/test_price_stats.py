"""Tests for CPT/HCPCS price distribution stats."""

from datetime import date
from pathlib import Path

import pandas as pd

from src.config import ScraperConfig, get_output_path, get_price_stats_path
from src.models import DataFormat, HospitalConfig, ScrapeResult, ScrapeStatus
from src.price_stats import (
    collect_price_file_paths,
    compute_price_stats,
    format_stats_preview,
    write_price_stats_csv,
    write_price_stats_for_run,
)


def _hospital(ccn: str = "340001", state: str = "TN") -> HospitalConfig:
    return HospitalConfig(
        ccn=ccn,
        hospital_npi=ccn.zfill(10),
        can_automate=True,
        hospital="Test Hospital",
        state=state,
        file_url="https://example.com/prices.json",
        type=DataFormat.JSON,
    )


def _write_jsonl(path: Path, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


class TestGetPriceStatsPath:
    def test_all_states(self, tmp_path: Path) -> None:
        config = ScraperConfig(project_root=tmp_path)
        assert get_price_stats_path(config) == tmp_path / "data" / "outputs" / "all_price_stats.csv"

    def test_state_ccn_and_hcpcs(self, tmp_path: Path) -> None:
        config = ScraperConfig(project_root=tmp_path, hcpcs_codes=["99213", "J0585"])
        path = get_price_stats_path(config, state_filter="vt", ccn_filter="470011")
        assert (
            path
            == tmp_path / "data" / "outputs" / "VT_470011_hcpcs_99213_J0585_price_stats.csv"
        )


class TestComputePriceStats:
    def test_empty_paths(self) -> None:
        df = compute_price_stats([])
        assert df.empty
        assert list(df.columns) == [
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

    def test_known_distribution(self, tmp_path: Path) -> None:
        path = tmp_path / "TN" / "340001.jsonl"
        _write_jsonl(
            path,
            [
                '{"cpt":"99213","type":"cash","price":1}',
                '{"cpt":"99213","type":"cash","price":2}',
                '{"cpt":"99213","type":"cash","price":3}',
                '{"cpt":"99213","type":"cash","price":4}',
                '{"cpt":"99213","type":"cash","price":5}',
            ],
        )
        df = compute_price_stats(
            [(path, "TN")],
            concept_names={"99213": "Office visit"},
        )
        assert len(df) == 1
        row = df.iloc[0]
        assert row["drug_name"] == "Office visit"
        assert row["hcpcs_code"] == "99213"
        assert row["state"] == "TN"
        assert row["type"] == "cash"
        assert row["n"] == 5
        assert row["n_hospitals"] == 1
        assert row["min"] == 1.0
        assert row["p25"] == 2.0
        assert row["median"] == 3.0
        assert row["mean"] == 3.0
        assert row["p75"] == 4.0
        assert row["max"] == 5.0

    def test_splits_by_state_and_type(self, tmp_path: Path) -> None:
        vt = tmp_path / "VT" / "470011.jsonl"
        nh = tmp_path / "NH" / "300001.jsonl"
        _write_jsonl(vt, ['{"cpt":"J1303","type":"cash","price":10}'])
        _write_jsonl(nh, ['{"cpt":"J1303","type":"cash","price":30}'])
        df = compute_price_stats([(vt, "VT"), (nh, "NH")])
        assert len(df) == 2
        assert set(df["state"]) == {"VT", "NH"}
        assert set(df["hcpcs_code"]) == {"J1303"}

    def test_splits_by_type_and_skips_nonpositive(self, tmp_path: Path) -> None:
        path = tmp_path / "340001.jsonl"
        _write_jsonl(
            path,
            [
                '{"cpt":"99213","type":"cash","price":10}',
                '{"cpt":"99213","type":"gross","price":20}',
                '{"cpt":"99213","type":"net","price":15}',
                '{"cpt":"99213","type":"cash","price":0}',
                '{"cpt":"99213","type":"cash","price":-5}',
                "not json",
            ],
        )
        df = compute_price_stats([(path, "TN")])
        assert list(df["type"]) == ["cash", "gross", "net"]
        cash = df[df["type"] == "cash"].iloc[0]
        assert cash["n"] == 1
        assert cash["min"] == 10.0

    def test_multiple_hospitals_same_state(self, tmp_path: Path) -> None:
        a = tmp_path / "340001.jsonl"
        b = tmp_path / "340002.jsonl"
        _write_jsonl(a, ['{"cpt":"99213","type":"cash","price":10}'])
        _write_jsonl(b, ['{"cpt":"99213","type":"cash","price":30}'])
        df = compute_price_stats([(a, "TN"), (b, "TN")])
        row = df.iloc[0]
        assert row["n"] == 2
        assert row["n_hospitals"] == 2
        assert row["state"] == "TN"
        assert row["min"] == 10.0
        assert row["max"] == 30.0
        assert row["mean"] == 20.0
        assert row["median"] == 20.0


class TestWritePriceStatsForRun:
    def test_uses_success_and_skipped_files(self, tmp_path: Path) -> None:
        config = ScraperConfig(project_root=tmp_path, data_dir=tmp_path / "data")
        hospital = _hospital()
        path = get_output_path(config, hospital)
        _write_jsonl(
            path,
            [
                '{"cpt":"99213","type":"cash","price":89.5}',
                '{"cpt":"99213","type":"gross","price":150}',
            ],
        )
        result = ScrapeResult(
            scrape_date=date.today(),
            hospital_npi=hospital.hospital_npi,
            ccn=hospital.ccn,
            status=ScrapeStatus.SUCCESS,
            file_url=hospital.file_url,
            records_scraped=2,
        )
        stats_path, df = write_price_stats_for_run(
            config, [(hospital, result)], state_filter="TN", ccn_filter="340001"
        )
        assert stats_path == tmp_path / "data" / "outputs" / "TN_340001_price_stats.csv"
        assert stats_path.exists()
        assert len(df) == 2
        written = pd.read_csv(stats_path)
        assert list(written.columns)[:4] == ["drug_name", "hcpcs_code", "state", "type"]
        assert set(written["state"]) == {"TN"}

    def test_skips_failed_hospitals(self, tmp_path: Path) -> None:
        config = ScraperConfig(project_root=tmp_path, data_dir=tmp_path / "data")
        hospital = _hospital()
        path = get_output_path(config, hospital)
        _write_jsonl(path, ['{"cpt":"99213","type":"cash","price":10}'])
        result = ScrapeResult.failure(
            hospital_npi=hospital.hospital_npi,
            file_url=hospital.file_url,
            error=RuntimeError("boom"),
            duration_seconds=1.0,
            ccn=hospital.ccn,
        )
        paths = collect_price_file_paths(config, [(hospital, result)])
        assert paths == []

    def test_preview_mentions_percentiles(self, tmp_path: Path) -> None:
        path = tmp_path / "out.csv"
        df = pd.DataFrame(
            [
                {
                    "drug_name": "Injection, ravulizumab-cwvz, 10 mg",
                    "hcpcs_code": "J1303",
                    "state": "VT",
                    "type": "cash",
                    "n": 1,
                    "n_hospitals": 1,
                    "min": 10.0,
                    "p25": 10.0,
                    "median": 10.0,
                    "mean": 10.0,
                    "p75": 10.0,
                    "max": 10.0,
                }
            ]
        )
        write_price_stats_csv(df, path)
        preview = format_stats_preview(df)
        assert "J1303" in preview
        assert "VT" in preview
        assert "p25" in preview
        assert "p75" in preview


def test_empty_preview() -> None:
    preview = format_stats_preview(pd.DataFrame(columns=["hcpcs_code"]))
    assert "No price observations" in preview
