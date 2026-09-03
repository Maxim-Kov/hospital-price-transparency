"""Tests for offline --from-existing HCPCS filtering."""

from pathlib import Path

from src.config import ScraperConfig, get_full_archive_path, get_output_path
from src.from_existing import (
    filter_jsonl_by_hcpcs,
    process_hospital_from_existing,
    run_from_existing,
)
from src.models import DataFormat, HospitalConfig, ScrapeStatus


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


class TestGetFullArchivePath:
    def test_ignores_hcpcs_filter(self, tmp_path: Path) -> None:
        config = ScraperConfig(
            project_root=tmp_path,
            data_dir=tmp_path / "data",
            hcpcs_codes=["J1303"],
        )
        hospital = _hospital()
        assert get_full_archive_path(config, hospital) == tmp_path / "data" / "TN" / "340001.jsonl"
        assert get_output_path(config, hospital) == (
            tmp_path / "data" / "outputs" / "J1303" / "TN" / "340001.jsonl"
        )


class TestFilterJsonlByHcpcs:
    def test_keeps_matching_rows(self, tmp_path: Path) -> None:
        source = tmp_path / "src.jsonl"
        dest = tmp_path / "out.jsonl"
        _write_jsonl(
            source,
            [
                '{"cpt":"99213","type":"cash","price":10}',
                '{"cpt":"j1303","type":"gross","price":20}',
                '{"cpt":"J1303","type":"net","price":15,"payer":"aetna","plan":"other"}',
            ],
        )
        n = filter_jsonl_by_hcpcs(source, dest, {"J1303"})
        assert n == 2
        text = dest.read_text(encoding="utf-8")
        assert "J1303" in text
        assert "99213" not in text

    def test_dry_run_does_not_write(self, tmp_path: Path) -> None:
        source = tmp_path / "src.jsonl"
        dest = tmp_path / "out.jsonl"
        _write_jsonl(source, ['{"cpt":"J1303","type":"cash","price":10}'])
        n = filter_jsonl_by_hcpcs(source, dest, {"J1303"}, dry_run=True)
        assert n == 1
        assert not dest.exists()

    def test_zero_matches_removes_stale_dest(self, tmp_path: Path) -> None:
        source = tmp_path / "src.jsonl"
        dest = tmp_path / "out.jsonl"
        _write_jsonl(source, ['{"cpt":"99213","type":"cash","price":10}'])
        dest.write_text('{"cpt":"J1303","type":"cash","price":1}\n', encoding="utf-8")
        n = filter_jsonl_by_hcpcs(source, dest, {"J1303"})
        assert n == 0
        assert not dest.exists()


class TestProcessHospitalFromExisting:
    def test_success_from_archive(self, tmp_path: Path) -> None:
        config = ScraperConfig(
            project_root=tmp_path,
            data_dir=tmp_path / "data",
            hcpcs_codes=["J1303"],
        )
        hospital = _hospital()
        archive = get_full_archive_path(config, hospital)
        _write_jsonl(
            archive,
            [
                '{"cpt":"J1303","type":"cash","price":100}',
                '{"cpt":"99213","type":"cash","price":50}',
            ],
        )
        hospital, result, message = process_hospital_from_existing(hospital, config)
        assert result.status == ScrapeStatus.SUCCESS
        assert result.records_scraped == 1
        assert "Filtered" in message
        out = get_output_path(config, hospital)
        assert out.exists()
        assert "J1303" in out.read_text(encoding="utf-8")
        assert "99213" not in out.read_text(encoding="utf-8")

    def test_skips_missing_archive(self, tmp_path: Path) -> None:
        config = ScraperConfig(
            project_root=tmp_path,
            data_dir=tmp_path / "data",
            hcpcs_codes=["J1303"],
        )
        hospital = _hospital()
        hospital, result, message = process_hospital_from_existing(hospital, config)
        assert result.status == ScrapeStatus.SKIPPED
        assert "no archive" in message.lower() or "Skipped" in message

    def test_run_from_existing_batch(self, tmp_path: Path) -> None:
        config = ScraperConfig(
            project_root=tmp_path,
            data_dir=tmp_path / "data",
            hcpcs_codes=["J1303"],
        )
        a = _hospital("340001")
        b = _hospital("340002")
        _write_jsonl(
            get_full_archive_path(config, a),
            ['{"cpt":"J1303","type":"cash","price":10}'],
        )
        results = run_from_existing(config, [a, b])
        assert len(results) == 2
        assert results[0][1].status == ScrapeStatus.SUCCESS
        assert results[1][1].status == ScrapeStatus.SKIPPED

    def test_run_from_existing_parallel_by_state(self, tmp_path: Path) -> None:
        config = ScraperConfig(
            project_root=tmp_path,
            data_dir=tmp_path / "data",
            hcpcs_codes=["J1303"],
        )
        tn = _hospital("340001", state="TN")
        vt = _hospital("470011", state="VT")
        _write_jsonl(
            get_full_archive_path(config, tn),
            ['{"cpt":"J1303","type":"cash","price":10}'],
        )
        _write_jsonl(
            get_full_archive_path(config, vt),
            ['{"cpt":"J1303","type":"cash","price":20}'],
        )
        results = run_from_existing(config, [tn, vt], parallel=2)
        assert len(results) == 2
        assert all(r[1].status == ScrapeStatus.SUCCESS for r in results)
        assert {r[0].state for r in results} == {"TN", "VT"}
