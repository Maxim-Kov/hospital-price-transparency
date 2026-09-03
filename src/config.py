"""Configuration management for the scraper.

Handles loading hospital configuration from URL JSON files and provides
centralized access to paths and settings. Uses CCN-based identification
with state-organized directory structure.
"""

import json
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, Field, field_validator

from .models import DataFormat, HospitalConfig
from .utils.logger import get_logger

logger = get_logger(__name__)

# Valid US state codes
VALID_STATES = {
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
    "MP",  # Territories
}


class ScraperConfig(BaseModel):
    """Global scraper configuration."""

    # Paths
    project_root: Path = Field(default_factory=lambda: Path(__file__).parent.parent)
    data_dir: Path | None = Field(default=None)
    dim_dir: Path | None = Field(default=None)
    logs_dir: Path | None = Field(default=None)

    # HTTP settings
    http_timeout: int = Field(default=60, description="HTTP request timeout in seconds")
    max_retries: int = Field(default=3, description="Maximum retry attempts")

    # Logging settings
    log_level: str = Field(default="INFO", description="Logging level")
    json_logs: bool = Field(default=False, description="Use JSON log format")

    # Optional post-parse filter; when set, output is written under data/outputs/
    hcpcs_codes: list[str] = Field(
        default_factory=list,
        description="If set, keep only these CPT/HCPCS codes in scrape output",
    )

    @field_validator("hcpcs_codes")
    @classmethod
    def uppercase_hcpcs_codes(cls, v: list[str]) -> list[str]:
        """Normalize filter codes to uppercase."""
        return [c.strip().upper() for c in v if c and str(c).strip()]

    def model_post_init(self, __context: object) -> None:
        """Set default paths relative to project root."""
        if self.data_dir is None:
            self.data_dir = self.project_root / "data"
        if self.dim_dir is None:
            self.dim_dir = self.project_root / "dim"
        if self.logs_dir is None:
            self.logs_dir = self.project_root / "logs"

    @property
    def urls_dir(self) -> Path:
        """Path to the directory containing state URL JSON files."""
        assert self.dim_dir is not None  # Set in model_post_init
        return self.dim_dir / "urls"

    @property
    def status_dir(self) -> Path:
        """Path to the directory for per-state status files."""
        return self.project_root / "status"

    @property
    def outputs_dir(self) -> Path:
        """Path for search/filter outputs (HCPCS extracts and price stats)."""
        assert self.data_dir is not None  # Set in model_post_init
        return self.data_dir / "outputs"

    @property
    def concept_csv_path(self) -> Path:
        """Path to CONCEPT.csv.gz (OHDSI Athena vocabulary)."""
        assert self.dim_dir is not None  # Set in model_post_init
        return self.dim_dir / "CONCEPT.csv.gz"


def _parse_row_to_config(row_dict: dict) -> HospitalConfig:
    """Parse a row dict into a HospitalConfig.

    Handles type conversion, NaN values, and field normalization.
    """
    # Handle NaN values
    for key, value in list(row_dict.items()):
        if pd.isna(value):
            row_dict[key] = None

    # Handle type conversion for payer field
    if row_dict.get("payer") in ["TRUE", "FALSE", True, False]:
        row_dict["payer"] = row_dict["payer"] in ["TRUE", True]

    # Handle can_automate boolean
    if row_dict.get("can_automate") in ["TRUE", "FALSE", True, False]:
        row_dict["can_automate"] = row_dict["can_automate"] in ["TRUE", True]
    elif row_dict.get("can_automate") is None:
        row_dict["can_automate"] = False

    # Handle data format enum
    if row_dict.get("type"):
        try:
            row_dict["type"] = DataFormat(row_dict["type"])
        except ValueError:
            row_dict["type"] = None

    # Convert NPI to string if present
    if row_dict.get("hospital_npi") is not None:
        row_dict["hospital_npi"] = str(int(row_dict["hospital_npi"]))

    # Handle skiprow as int
    if row_dict.get("skiprow") is not None:
        row_dict["skiprow"] = int(row_dict["skiprow"])

    return HospitalConfig(**row_dict)


def _detect_format_from_url(url: str) -> DataFormat | None:
    """Detect data format from URL extension or patterns.

    Uses multiple strategies:
    1. File extensions (.json, .csv, .xlsx, etc.)
    2. Known provider URL patterns
    3. Common API endpoint patterns

    Args:
        url: File URL

    Returns:
        DataFormat enum or None if unknown
    """
    url_lower = url.lower()

    # 1. Check file extensions (most reliable)
    if ".json" in url_lower:
        return DataFormat.JSON
    elif ".csv" in url_lower:
        return DataFormat.CSV
    elif ".xlsx" in url_lower or ".xls" in url_lower:
        return DataFormat.XLSX
    elif ".xml" in url_lower:
        return DataFormat.XML
    elif ".zip" in url_lower:
        return DataFormat.ZIP
    elif ".ashx" in url_lower:
        # ASP.NET handlers (Yale-New Haven, Bon Secours, St. Luke's, Mercy)
        # These typically serve CSV content based on filename patterns
        return DataFormat.CSV

    # 2. Check known provider URL patterns that indicate format
    # ClaraPrice machine-readable endpoints -> JSON
    if "claraprice.net" in url_lower and "machine-readable" in url_lower:
        return DataFormat.JSON

    # Craneware API -> CSV (despite the API endpoint)
    if "craneware.com/api-pricing-transparency" in url_lower:
        return DataFormat.CSV

    # Panacea/Trinity MRF downloads -> ZIP (contains CSV)
    if "panaceainc.com/mrfdownload" in url_lower:
        return DataFormat.ZIP

    # Hospital Price Index blob storage -> typically CSV
    if "sthpiprd.blob.core.windows.net" in url_lower:
        return DataFormat.CSV

    # AccuReg price transparency -> CSV
    if "pricetransparency.accureg.net" in url_lower:
        return DataFormat.CSV

    # 3. Generic patterns - be more conservative, don't assume JSON
    # Only use JSON for explicit machine-readable if no other match
    if "standardcharges" in url_lower and "." not in url_lower.split("/")[-1]:
        # CMS standard naming without extension - could be JSON or CSV
        # Return None to let content-type detection handle it
        return None

    return None


def load_hospital_configs_from_urls(
    config: ScraperConfig,
    state_filter: str | None = None,
    ccn_filter: str | None = None,
) -> list[HospitalConfig]:
    """Load hospital configurations from URL JSON files (dim/urls/*.json).

    These JSON files are created by the browser-based scraper from
    hospitalpricingfiles.org and contain CCN, hospital name, and file URLs.

    Args:
        config: Scraper configuration
        state_filter: If provided, only load from this state's JSON file
        ccn_filter: If provided, only return the hospital with this CCN

    Returns:
        List of validated HospitalConfig objects
    """
    urls_dir = config.urls_dir
    if not urls_dir.exists():
        logger.warning("urls_dir_not_found", path=str(urls_dir))
        return []

    configs = []

    # Find JSON files to process
    if state_filter:
        json_files = [urls_dir / f"{state_filter.lower()}.json"]
    else:
        json_files = list(urls_dir.glob("*.json"))

    for json_file in json_files:
        if not json_file.exists():
            logger.warning("state_json_not_found", path=str(json_file))
            continue

        # Extract state from filename (e.g., vt.json -> VT)
        state = json_file.stem.upper()

        # Skip non-state files (e.g., needs_review.json)
        if state not in VALID_STATES:
            continue

        try:
            with open(json_file) as f:
                hospitals = json.load(f)
        except Exception as e:
            logger.error("json_load_error", file=str(json_file), error=str(e))
            continue

        logger.info("loading_state_urls", state=state, count=len(hospitals))

        for entry in hospitals:
            ccn = entry.get("ccn", "")
            file_url = entry.get("file_url", "")

            # Skip if no CCN (required for file organization)
            if not ccn:
                continue

            # Skip if no file URL
            if not file_url:
                continue

            # Apply CCN filter
            if ccn_filter and ccn.upper() != ccn_filter.upper():
                continue

            # Detect format from URL
            data_format = _detect_format_from_url(file_url)

            # Create a config dict matching HospitalConfig fields
            config_dict = {
                "ccn": ccn,
                "hospital_npi": ccn.zfill(10),  # Placeholder NPI from CCN
                "can_automate": True,
                "hospital": entry.get("hospital_name", ""),
                "address": entry.get("address", ""),
                "state": state,
                "file_url": file_url,
                "parent_url": entry.get("transparency_page", ""),
                "type": data_format,
                "idn": "",  # Unknown from this source
            }

            try:
                hospital_config = HospitalConfig(**config_dict)
                configs.append(hospital_config)
            except Exception as e:
                logger.warning(
                    "url_config_parse_error",
                    ccn=ccn,
                    hospital=entry.get("hospital_name", "unknown"),
                    error=str(e),
                )

    logger.info(
        "loaded_url_configs",
        total=len(configs),
        states=len(json_files),
        state_filter=state_filter,
    )

    return configs


def load_concept_codes(config: ScraperConfig) -> pd.DataFrame:
    """Load OHDSI Athena CPT4 and HCPCS concept codes.

    Args:
        config: Scraper configuration

    Returns:
        DataFrame with concept_code column for valid CPT4 and HCPCS codes
    """
    df = pd.read_csv(
        config.concept_csv_path,
        compression="gzip",
        sep="\t",
    )
    # Filter to CPT4 and HCPCS vocabularies
    df = df[df["vocabulary_id"].isin(["CPT4", "HCPCS"])]
    return df[["concept_code"]]


def get_output_path(config: ScraperConfig, hospital: HospitalConfig) -> Path:
    """Get the output path for a hospital's JSONL file.

    Full scrapes use data/{STATE}/{CCN}.jsonl.
    HCPCS-filtered searches use data/outputs/{CODES}/{STATE}/{CCN}.jsonl so
    they do not overwrite the full-hospital git archive.

    Args:
        config: Scraper configuration
        hospital: Hospital configuration

    Returns:
        Path to the output JSONL file
    """
    assert config.data_dir is not None  # Set in model_post_init
    if config.hcpcs_codes:
        data_dir = config.outputs_dir / "_".join(config.hcpcs_codes)
    else:
        data_dir = config.data_dir

    if hospital.state and hospital.ccn:
        state_dir = data_dir / hospital.state.upper()
        state_dir.mkdir(parents=True, exist_ok=True)
        return state_dir / f"{hospital.ccn}.jsonl"

    # Fallback for hospitals without CCN (shouldn't happen with new system)
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir / f"{hospital.identifier}.jsonl"


def get_full_archive_path(config: ScraperConfig, hospital: HospitalConfig) -> Path:
    """Full-hospital archive path data/{STATE}/{CCN}.jsonl.

    Ignores --hcpcs so offline filters can read the git-scraped snapshot.
    Does not create directories.
    """
    assert config.data_dir is not None  # Set in model_post_init
    if hospital.state and hospital.ccn:
        return config.data_dir / hospital.state.upper() / f"{hospital.ccn}.jsonl"
    return config.data_dir / f"{hospital.identifier}.jsonl"


def get_price_stats_path(
    config: ScraperConfig,
    state_filter: str | None = None,
    ccn_filter: str | None = None,
) -> Path:
    """CSV path for per-run CPT/HCPCS price distribution stats.

    Written under data/outputs/ so search results stay together.
    """
    outputs_dir = config.outputs_dir
    outputs_dir.mkdir(parents=True, exist_ok=True)

    parts: list[str] = []
    if state_filter:
        parts.append(state_filter.strip().upper())
    if ccn_filter:
        parts.append(ccn_filter.strip().upper())
    if config.hcpcs_codes:
        parts.append("hcpcs_" + "_".join(config.hcpcs_codes))
    stem = "_".join(parts) if parts else "all"
    return outputs_dir / f"{stem}_price_stats.csv"


def get_data_age_days(config: ScraperConfig, hospital: HospitalConfig) -> float | None:
    """Get the age of existing data file in days.

    Used for incremental scraping to skip hospitals with recent data.

    Args:
        config: Scraper configuration
        hospital: Hospital configuration

    Returns:
        Age in days as a float, or None if no data file exists
    """
    from datetime import datetime

    output_path = get_output_path(config, hospital)
    if not output_path.exists():
        return None

    mtime = datetime.fromtimestamp(output_path.stat().st_mtime)
    age_seconds = (datetime.now() - mtime).total_seconds()
    return age_seconds / 86400  # Convert to days
