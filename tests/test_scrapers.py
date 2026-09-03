"""Tests for scraper implementations."""

from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.config import ScraperConfig, get_output_path
from src.models import DataFormat, HospitalConfig
from src.normalizers import CPTNormalizer
from src.scrapers.cms_csv_scraper import CMSStandardCSVScraper
from src.scrapers.cms_json_scraper import CMSStandardJSONScraper, HyveCMSJSONScraper
from src.scrapers.registry import ScraperRegistry, get_scraper
from src.utils.http_client import RetryHTTPClient


@pytest.fixture
def mock_http_client():
    """Create a mock HTTP client."""
    return MagicMock(spec=RetryHTTPClient)


@pytest.fixture
def mock_normalizer():
    """Create a mock CPT normalizer."""
    normalizer = MagicMock(spec=CPTNormalizer)
    # Make normalize return input unchanged for testing
    normalizer.normalize.side_effect = lambda df, **kwargs: df
    normalizer.filter_by_hcpcs.side_effect = CPTNormalizer.filter_by_hcpcs
    return normalizer


@pytest.fixture
def scraper_config():
    """Create a test scraper config."""
    return ScraperConfig()


@pytest.fixture
def json_hospital_config():
    """Create a JSON hospital config for testing."""
    return HospitalConfig(
        hospital_npi="1234567890",
        ccn="340001",
        can_automate=True,
        idn="Parkridge",
        hospital="Test Hospital",
        address="123 Test St",
        cbsa=12345,
        cbsa_title="Test City",
        state="TN",
        parent_url="https://example.com",
        file_url="https://example.com/prices.json",
        type=DataFormat.JSON,
    )


@pytest.fixture
def json_covenant_config():
    """Create a Covenant Health JSON config for testing."""
    return HospitalConfig(
        hospital_npi="1234567890",
        ccn="340002",
        can_automate=True,
        idn="Covenant Health",
        hospital="Test Covenant Hospital",
        address="123 Test St",
        cbsa=12345,
        cbsa_title="Test City",
        state="TN",
        parent_url="https://example.com",
        file_url="https://example.com/prices.json",
        type=DataFormat.JSON,
    )


@pytest.fixture
def csv_hospital_config():
    """Create a CSV hospital config for testing."""
    return HospitalConfig(
        hospital_npi="1234567890",
        ccn="340003",
        can_automate=True,
        idn="Unknown",
        hospital="Test CSV Hospital",
        address="123 Test St",
        cbsa=12345,
        cbsa_title="Test City",
        state="TN",
        parent_url="https://example.com",
        file_url="https://example.com/prices.csv",
        type=DataFormat.CSV,
    )


class TestCMSStandardJSONScraper:
    """Tests for CMSStandardJSONScraper."""

    def test_parse_data_cms_format(
        self, json_hospital_config, scraper_config, mock_http_client, mock_normalizer
    ):
        """Test CMS standard JSON parsing with standard_charge_information format."""
        scraper = CMSStandardJSONScraper(
            hospital_config=json_hospital_config,
            scraper_config=scraper_config,
            http_client=mock_http_client,
            normalizer=mock_normalizer,
        )

        json_data = {
            "standard_charge_information": [
                {
                    "billing_code_information": [{"type": "CPT", "code": "99213"}],
                    "standard_charges": [{"gross_charge": 100, "discounted_cash": 80}],
                },
                {
                    "billing_code_information": [{"type": "CPT", "code": "99214"}],
                    "standard_charges": [{"gross_charge": 150, "discounted_cash": 120}],
                },
            ]
        }

        df = scraper.parse_data(json_data)

        assert len(df) == 2
        assert df.iloc[0]["concept_code"] == "99213"
        assert df.iloc[0]["gross"] == 100
        assert df.iloc[0]["cash"] == 80

    def test_parse_data_with_payers_information(
        self, json_hospital_config, scraper_config, mock_http_client, mock_normalizer
    ):
        """Test JSON parsing extracts per-payer negotiated nets."""
        scraper = CMSStandardJSONScraper(
            hospital_config=json_hospital_config,
            scraper_config=scraper_config,
            http_client=mock_http_client,
            normalizer=mock_normalizer,
        )

        json_data = {
            "standard_charge_information": [
                {
                    "code_information": [{"type": "CPT", "code": "80320"}],
                    "standard_charges": [
                        {
                            "gross_charge": 76.55,
                            "discounted_cash": 48.23,
                            "payers_information": [
                                {
                                    "payer_name": "AETNA",
                                    "plan_name": "COMMERCIAL",
                                    "standard_charge_dollar": 26.79,
                                },
                                {
                                    "payer_name": "KANCARE AETNA",
                                    "plan_name": "MEDICAID ADVANTAGE KANCARE AETNA",
                                    "standard_charge_dollar": 15.5,
                                },
                                {
                                    "payer_name": "CDM DEFAULT",
                                    "plan_name": "CDM DEFAULT",
                                    "estimated_amount": 999999999,
                                },
                            ],
                        }
                    ],
                },
            ]
        }

        df = scraper.parse_data(json_data)

        overall = df[df["net"].isna()]
        nets = df[df["net"].notna()]
        assert len(overall) == 1
        assert overall.iloc[0]["gross"] == 76.55
        assert overall.iloc[0]["cash"] == 48.23
        assert len(nets) == 2  # sentinel CDM row skipped (no usable dollar)
        assert set(nets["payer_raw"]) == {"AETNA", "KANCARE AETNA"}
        assert 26.79 in nets["net"].values
        assert 15.5 in nets["net"].values

    def test_parse_data_flat_list(
        self, json_hospital_config, scraper_config, mock_http_client, mock_normalizer
    ):
        """Test CMS JSON parsing with flat list format (direct charges array)."""
        scraper = CMSStandardJSONScraper(
            hospital_config=json_hospital_config,
            scraper_config=scraper_config,
            http_client=mock_http_client,
            normalizer=mock_normalizer,
        )

        # Some hospitals use a flat list format
        json_data = [
            {
                "code": "99213",
                "type": "CPT",
                "description": "Office Visit",
                "gross_charge": 100,
                "discounted_cash": 80,
            },
            {
                "code": "99214",
                "type": "CPT",
                "description": "Extended Visit",
                "gross_charge": 150,
                "discounted_cash": 120,
            },
        ]

        df = scraper.parse_data(json_data)

        assert len(df) == 2
        assert df.iloc[0]["concept_code"] == "99213"


class TestHyveCMSJSONScraper:
    """Tests for HyveCMSJSONScraper (Covenant Health format).

    HyveCMSJSONScraper inherits from CMSStandardJSONScraper without modifications,
    so we just test that it can be instantiated and uses the same parsing.
    """

    def test_inherits_cms_standard(
        self, json_covenant_config, scraper_config, mock_http_client, mock_normalizer
    ):
        """Test that HyveCMSJSONScraper inherits from CMSStandardJSONScraper."""
        scraper = HyveCMSJSONScraper(
            hospital_config=json_covenant_config,
            scraper_config=scraper_config,
            http_client=mock_http_client,
            normalizer=mock_normalizer,
        )

        assert isinstance(scraper, CMSStandardJSONScraper)

    def test_parse_data_uses_cms_format(
        self, json_covenant_config, scraper_config, mock_http_client, mock_normalizer
    ):
        """Test HyveCMSJSONScraper uses CMS standard format parsing."""
        scraper = HyveCMSJSONScraper(
            hospital_config=json_covenant_config,
            scraper_config=scraper_config,
            http_client=mock_http_client,
            normalizer=mock_normalizer,
        )

        # Use CMS standard format
        json_data = {
            "standard_charge_information": [
                {
                    "billing_code_information": [{"type": "CPT", "code": "99213"}],
                    "standard_charges": [{"gross_charge": 100, "discounted_cash": 80}],
                },
            ]
        }

        df = scraper.parse_data(json_data)

        assert len(df) == 1
        assert df.iloc[0]["concept_code"] == "99213"


class TestCMSStandardCSVScraper:
    """Tests for CMSStandardCSVScraper."""

    def test_parse_data(
        self, csv_hospital_config, scraper_config, mock_http_client, mock_normalizer
    ):
        """Test CMS standard CSV parsing with pipe-delimited code columns."""
        scraper = CMSStandardCSVScraper(
            hospital_config=csv_hospital_config,
            scraper_config=scraper_config,
            http_client=mock_http_client,
            normalizer=mock_normalizer,
        )

        # CMS 2.0 CSV format: 2 header rows, then data
        csv_data = """Hospital Name,Test CSV Hospital,,,,,,
Hospital Address,123 Test St,,,,,,
description,code|1,code|1|type,code|2,code|2|type,standard_charge|gross,standard_charge|discounted_cash,notes
Office Visit,99213,CPT,G0001,HCPCS,100.00,80.00,
Extended Visit,99214,CPT,,,150.00,120.00,
"""

        df = scraper.parse_data(csv_data)

        assert len(df) >= 1
        # The parser should find CPT codes
        assert any(df["concept_code"] == "99213")

    def test_parse_discards_unrelated_hcpcs_rows(
        self, csv_hospital_config, mock_http_client, mock_normalizer
    ):
        """Parse-time --hcpcs filter drops other codes on the same and other rows."""
        config = ScraperConfig(hcpcs_codes=["99213"])
        scraper = CMSStandardCSVScraper(
            hospital_config=csv_hospital_config,
            scraper_config=config,
            http_client=mock_http_client,
            normalizer=mock_normalizer,
        )

        csv_data = """Hospital Name,Test CSV Hospital,,,,,,
Hospital Address,123 Test St,,,,,,
description,code|1,code|1|type,code|2,code|2|type,standard_charge|gross,standard_charge|discounted_cash,notes
Office Visit,99213,CPT,G0001,HCPCS,100.00,80.00,
Extended Visit,99214,CPT,,,150.00,120.00,
"""

        df = scraper.parse_data(csv_data)
        assert set(df["concept_code"].unique()) == {"99213"}

    def test_hcpcs_prefilter_skips_nonmatching_rows(
        self, csv_hospital_config, mock_http_client, mock_normalizer
    ):
        """Vectorized prefilter keeps padded and secondary-column matches only."""
        config = ScraperConfig(hcpcs_codes=["99213"])
        scraper = CMSStandardCSVScraper(
            hospital_config=csv_hospital_config,
            scraper_config=config,
            http_client=mock_http_client,
            normalizer=mock_normalizer,
        )
        df = pd.DataFrame(
            {
                "code|1": ["99214", "099213", "G0001"],
                "code|1|type": ["CPT", "CPT", "HCPCS"],
                "code|2": ["", "G0001", "99213"],
                "code|2|type": ["", "HCPCS", "CPT"],
                "standard_charge|gross": ["1", "2", "3"],
            }
        )
        filtered = scraper._filter_df_to_hcpcs(df)
        assert list(filtered["code|1"]) == ["099213", "G0001"]
        assert list(filtered["code|2"]) == ["G0001", "99213"]

    def test_parse_tall_payer_rows(
        self, csv_hospital_config, scraper_config, mock_http_client, mock_normalizer
    ):
        """Test tall CMS CSV format with payer_name / negotiated_dollar columns."""
        scraper = CMSStandardCSVScraper(
            hospital_config=csv_hospital_config,
            scraper_config=scraper_config,
            http_client=mock_http_client,
            normalizer=mock_normalizer,
        )

        csv_data = """hospital_name,last_updated_on
Test Hospital,01/01/2024
description,code|1,code|1|type,standard_charge|gross,standard_charge|discounted_cash,payer_name,plan_name,standard_charge|negotiated_dollar,estimated_amount
ETHANOL URINE,80320,CPT,76.55,48.23,CDM DEFAULT,CDM DEFAULT,,999999999
ETHANOL URINE,80320,CPT,76.55,,AETNA,COMMERCIAL,26.79,
ETHANOL URINE,80320,CPT,76.55,,KANCARE AETNA,MEDICAID ADVANTAGE KANCARE AETNA,15.5,
"""

        df = scraper.parse_data(csv_data)

        overall = df[df["net"].isna()]
        nets = df[df["net"].notna()]
        assert len(overall) == 1
        assert overall.iloc[0]["gross"] == 76.55
        assert overall.iloc[0]["cash"] == 48.23
        assert len(nets) == 2
        assert set(nets["payer_raw"]) == {"AETNA", "KANCARE AETNA"}

    def test_parse_wide_payer_columns(
        self, csv_hospital_config, scraper_config, mock_http_client, mock_normalizer
    ):
        """Test wide CMS CSV format with standard_charge|Payer|Plan columns."""
        scraper = CMSStandardCSVScraper(
            hospital_config=csv_hospital_config,
            scraper_config=scraper_config,
            http_client=mock_http_client,
            normalizer=mock_normalizer,
        )

        csv_data = """hospital_name,last_updated_on
Test Hospital,01/01/2024
description,code|1,code|1|type,standard_charge|gross,standard_charge|discounted_cash,standard_charge|Aetna|Commercial,standard_charge|United Healthcare|PPO
Office Visit,99213,CPT,150.00,100.00,90.00,85.00
"""

        df = scraper.parse_data(csv_data)

        overall = df[df["net"].isna()]
        nets = df[df["net"].notna()]
        assert len(overall) == 1
        assert len(nets) == 2
        assert set(nets["payer_raw"]) == {"Aetna", "United Healthcare"}
        assert set(nets["net"]) == {90.0, 85.0}


class TestScraperRegistry:
    """Tests for ScraperRegistry."""

    def test_get_idn_scraper_covenant(self, json_covenant_config):
        """Test getting IDN-specific scraper for Covenant Health."""
        scraper_class = ScraperRegistry.get_scraper_class(json_covenant_config)
        assert scraper_class == HyveCMSJSONScraper

    def test_get_idn_scraper_parkridge(self, json_hospital_config):
        """Test getting IDN-specific scraper for Parkridge."""
        scraper_class = ScraperRegistry.get_scraper_class(json_hospital_config)
        assert scraper_class == CMSStandardJSONScraper

    def test_get_format_scraper_json(self):
        """Test getting format-based scraper for JSON."""
        config = HospitalConfig(
            hospital_npi="1234567890",
            ccn="340099",
            can_automate=True,
            idn="Unknown",  # Not a registered IDN
            hospital="Test",
            address="Test",
            cbsa=12345,
            cbsa_title="Test",
            state="TN",
            parent_url="https://example.com",
            file_url="https://example.com/file.json",
            type=DataFormat.JSON,
        )
        scraper_class = ScraperRegistry.get_scraper_class(config)
        assert scraper_class == CMSStandardJSONScraper

    def test_get_format_scraper_csv(self, csv_hospital_config):
        """Test getting format-based scraper for CSV."""
        scraper_class = ScraperRegistry.get_scraper_class(csv_hospital_config)
        assert scraper_class == CMSStandardCSVScraper

    def test_get_url_provider_scraper(self):
        """Test URL pattern-based scraper selection."""
        config = HospitalConfig(
            hospital_npi="1234567890",
            ccn="340099",
            can_automate=True,
            idn="",
            hospital="Test",
            address="Test",
            cbsa=12345,
            cbsa_title="Test",
            state="TN",
            parent_url="https://example.com",
            # ClaraPrice URL pattern -> JSON scraper
            file_url="https://claraprice.net/machine-readable/hospital/123",
            type=None,  # Let URL pattern determine type
        )
        scraper_class = ScraperRegistry.get_scraper_class(config)
        assert scraper_class == CMSStandardJSONScraper

    def test_create_scraper(
        self, json_hospital_config, scraper_config, mock_http_client, mock_normalizer
    ):
        """Test creating a scraper instance."""
        scraper = ScraperRegistry.create_scraper(
            hospital_config=json_hospital_config,
            scraper_config=scraper_config,
            http_client=mock_http_client,
            normalizer=mock_normalizer,
        )

        assert scraper is not None
        assert isinstance(scraper, CMSStandardJSONScraper)

    def test_explicit_scraper_type(
        self, json_hospital_config, scraper_config, mock_http_client, mock_normalizer
    ):
        """Test explicit scraper_type override."""
        json_hospital_config.scraper_type = "HyveCMSJSONScraper"

        scraper = ScraperRegistry.create_scraper(
            hospital_config=json_hospital_config,
            scraper_config=scraper_config,
            http_client=mock_http_client,
            normalizer=mock_normalizer,
        )

        assert scraper is not None
        assert isinstance(scraper, HyveCMSJSONScraper)


class TestGetScraper:
    """Tests for get_scraper convenience function."""

    def test_get_scraper_returns_instance(
        self, json_hospital_config, scraper_config, mock_http_client, mock_normalizer
    ):
        """Test get_scraper returns a configured scraper instance."""
        scraper = get_scraper(
            hospital_config=json_hospital_config,
            scraper_config=scraper_config,
            http_client=mock_http_client,
            normalizer=mock_normalizer,
        )

        assert scraper is not None
        assert isinstance(scraper, CMSStandardJSONScraper)

    def test_get_scraper_returns_none_for_unknown_format(
        self, scraper_config, mock_http_client, mock_normalizer
    ):
        """Test get_scraper returns None for unknown format without IDN."""
        config = HospitalConfig(
            hospital_npi="1234567890",
            ccn="999999",
            can_automate=True,
            idn="Unknown",
            hospital="Test",
            address="Test",
            cbsa=12345,
            cbsa_title="Test",
            state="TN",
            parent_url="https://example.com",
            file_url="https://example.com/file.unknown",
            type=None,  # Unknown format
        )

        scraper = get_scraper(
            hospital_config=config,
            scraper_config=scraper_config,
            http_client=mock_http_client,
            normalizer=mock_normalizer,
        )

        assert scraper is None


class TestHcpcsFilter:
    """Tests for --hcpcs output path and post-parse filtering."""

    def test_output_path_full_scrape(self, json_hospital_config, tmp_path):
        config = ScraperConfig(project_root=tmp_path, data_dir=tmp_path / "data")
        path = get_output_path(config, json_hospital_config)
        assert path == tmp_path / "data" / "TN" / "340001.jsonl"

    def test_output_path_hcpcs_filter_does_not_clobber_full_snapshot(
        self, json_hospital_config, tmp_path
    ):
        config = ScraperConfig(
            project_root=tmp_path,
            data_dir=tmp_path / "data",
            hcpcs_codes=["99213", "J0585"],
        )
        path = get_output_path(config, json_hospital_config)
        assert path == tmp_path / "data" / "outputs" / "99213_J0585" / "TN" / "340001.jsonl"

    def test_normalize_applies_hcpcs_filter(
        self, json_hospital_config, mock_http_client
    ):
        config = ScraperConfig(hcpcs_codes=["99213"])
        normalizer = CPTNormalizer(concept_df=None)
        scraper = CMSStandardJSONScraper(
            hospital_config=json_hospital_config,
            scraper_config=config,
            http_client=mock_http_client,
            normalizer=normalizer,
        )

        df = pd.DataFrame(
            {
                "vocabulary_id": ["cpt", "cpt"],
                "concept_code": ["99213", "99214"],
                "gross": [100.0, 200.0],
                "cash": [80.0, 160.0],
            }
        )
        result = scraper.normalize(df)
        assert set(result["cpt"].unique()) == {"99213"}
        assert len(result) == 2

    def test_parse_discards_unrelated_hcpcs_rows(
        self, json_hospital_config, mock_http_client, mock_normalizer
    ):
        """Parse-time --hcpcs filter drops other codes and their payer nets."""
        config = ScraperConfig(hcpcs_codes=["99213"])
        scraper = CMSStandardJSONScraper(
            hospital_config=json_hospital_config,
            scraper_config=config,
            http_client=mock_http_client,
            normalizer=mock_normalizer,
        )

        json_data = {
            "standard_charge_information": [
                {
                    "code_information": [{"type": "CPT", "code": "99213"}],
                    "standard_charges": [
                        {
                            "gross_charge": 100,
                            "discounted_cash": 80,
                            "payers_information": [
                                {
                                    "payer_name": "AETNA",
                                    "plan_name": "COMMERCIAL",
                                    "standard_charge_dollar": 26.79,
                                }
                            ],
                        }
                    ],
                },
                {
                    "code_information": [{"type": "CPT", "code": "99214"}],
                    "standard_charges": [
                        {
                            "gross_charge": 200,
                            "discounted_cash": 160,
                            "payers_information": [
                                {
                                    "payer_name": "AETNA",
                                    "plan_name": "COMMERCIAL",
                                    "standard_charge_dollar": 50.0,
                                }
                            ],
                        }
                    ],
                },
            ]
        }

        df = scraper.parse_data(json_data)
        assert set(df["concept_code"].unique()) == {"99213"}
        assert df["gross"].dropna().tolist() == [100]
        nets = df[df["net"].notna()]
        assert len(nets) == 1
        assert nets.iloc[0]["net"] == 26.79
