"""CPT code normalization logic.

Extracts and modernizes the cleanup_charges() function from the original scrape.py.
Uses OHDSI Athena vocabulary for validation.
"""

import re
from pathlib import Path

import pandas as pd

from ..utils.logger import get_logger

logger = get_logger(__name__)

# Sentinel values used by some hospitals instead of a real dollar amount
SENTINEL_AMOUNTS = {999999999.0, 99999999.0, 9999999.0}


class CPTNormalizer:
    """Normalizes hospital price data to standard CPT/HCPCS schema.

    Handles:
    - Code cleaning (remove leading zeros, validate format)
    - Price column normalization (remove $, commas, convert to numeric)
    - Filtering to valid CPT4 and HCPCS codes using OHDSI Athena vocabulary
    - Melting wide format to long format (cpt, type, price [, payer, plan])
    """

    # Pattern for valid CPT codes (5 alphanumeric characters)
    CPT_PATTERN = re.compile(r"^[0-9A-Z]{5}$")

    def __init__(
        self,
        concept_df: pd.DataFrame | None = None,
        code_filter: list[str] | None = None,
    ):
        """Initialize the normalizer.

        Args:
            concept_df: DataFrame with 'concept_code' column containing valid CPT4 codes.
                       If None, validation against Athena vocabulary is skipped.
            code_filter: If set, keep only these CPT/HCPCS codes.
        """
        self.concept_codes: set[str] = set()
        if concept_df is not None:
            self.concept_codes = set(concept_df["concept_code"].astype(str).str.strip())
            logger.info("loaded_concept_codes", count=len(self.concept_codes))
        self.code_filter: set[str] = {
            c.strip().upper() for c in (code_filter or []) if c and str(c).strip()
        }

    @classmethod
    def from_file(cls, concept_path: Path) -> "CPTNormalizer":
        """Create a normalizer from the OHDSI Athena CONCEPT.csv.gz file.

        Args:
            concept_path: Path to CONCEPT.csv.gz

        Returns:
            Initialized CPTNormalizer
        """
        df = pd.read_csv(concept_path, compression="gzip", sep="\t")
        # Load both CPT4 and HCPCS vocabularies
        df = df[df["vocabulary_id"].isin(["CPT4", "HCPCS"])]
        return cls(df[["concept_code"]])

    @staticmethod
    def strip_leading_zero(code: str) -> str:
        """Remove leading zero from 6-character codes.

        Some data sources pad CPT codes with a leading zero.
        E.g., "099213" -> "99213"

        Args:
            code: The CPT code to clean

        Returns:
            Cleaned CPT code
        """
        code = str(code).strip()
        if len(code) == 6 and code[0] == "0":
            return code[1:]
        return code

    @staticmethod
    def clean_price(value: str | float | int) -> float | None:
        """Clean price value by removing currency symbols and commas.

        Args:
            value: Price value (may contain $, commas, etc.)

        Returns:
            Float price or None if invalid / sentinel
        """
        if pd.isna(value):
            return None

        if isinstance(value, (int, float)):
            price = float(value)
        else:
            # Remove currency symbols and commas
            cleaned = str(value).replace(",", "").replace("$", "").strip()
            try:
                price = float(cleaned)
            except (ValueError, TypeError):
                return None

        if price in SENTINEL_AMOUNTS:
            return None
        return price

    def _filter_vocab(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter to CPT/HCPCS codes and optional Athena vocabulary."""
        df = df.copy()
        df["vocabulary_id"] = df["vocabulary_id"].astype(str).str.lower()
        df = df[df["vocabulary_id"].isin(["cpt", "cpt4", "hcpcs"])]

        if self.concept_codes:
            initial_count = len(df)
            df = pd.merge(
                df,
                pd.DataFrame({"concept_code": list(self.concept_codes)}),
                on="concept_code",
                how="inner",
            )
            filtered_count = initial_count - len(df)
            if filtered_count > 0:
                logger.debug("filtered_invalid_codes", count=filtered_count)

        if self.code_filter:
            before = len(df)
            df = df[df["concept_code"].astype(str).str.upper().isin(self.code_filter)]
            dropped = before - len(df)
            if dropped > 0:
                logger.debug(
                    "filtered_unrelated_hcpcs",
                    dropped=dropped,
                    kept=sorted(self.code_filter),
                )

        return df

    def _finalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop invalid prices/codes and sort output."""
        df = df.drop_duplicates()
        df = df.dropna(subset=["price"])
        df = df[df["price"] > 0]
        df["price"] = df["price"].round(2)

        valid_mask = df["cpt"].apply(lambda x: bool(self.CPT_PATTERN.match(str(x))))
        invalid_count = (~valid_mask).sum()
        if invalid_count > 0:
            logger.warning("invalid_cpt_format", count=invalid_count)
            df = df[valid_mask]

        sort_cols = ["cpt", "type"]
        if "payer" in df.columns:
            sort_cols.extend(["payer", "plan"])
        df = df.sort_values(sort_cols)
        return df.reset_index(drop=True)

    @staticmethod
    def filter_by_hcpcs(df: pd.DataFrame, codes: list[str]) -> pd.DataFrame:
        """Drop rows whose procedure code is not in the given CPT/HCPCS set.

        Works on parsed frames (`concept_code`) and normalized frames (`cpt`).
        Leading-zero-padded codes are treated as the 5-character form.
        """
        if not codes or df.empty:
            return df
        col = (
            "concept_code"
            if "concept_code" in df.columns
            else ("cpt" if "cpt" in df.columns else None)
        )
        if col is None:
            return df
        wanted = {c.strip().upper() for c in codes}
        cleaned = (
            df[col]
            .astype(str)
            .map(CPTNormalizer.strip_leading_zero)
            .str.strip()
            .str.upper()
        )
        return df[cleaned.isin(wanted)].reset_index(drop=True)

    def normalize(
        self,
        df: pd.DataFrame,
        rename: bool = False,
        gross_col: str | None = None,
        cash_col: str | None = None,
        cpt_col: str | None = None,
    ) -> pd.DataFrame:
        """Normalize a DataFrame to the standard output schema.

        Expected input columns (after optional renaming):
        - vocabulary_id: Code type ('cpt', 'CPT4', etc.)
        - concept_code: The procedure code
        - gross: Gross charge amount
        - cash: Cash/discounted price
        - net (optional): Negotiated insurer dollar amount
        - payer (optional): Canonical payer id for net rows
        - plan (optional): Canonical plan id for net rows

        Output schema:
        - cpt: 5-character CPT code
        - type: 'gross', 'cash', or 'net'
        - price: Numeric price value
        - payer / plan: present on net rows (NaN on cash/gross)

        Args:
            df: Input DataFrame with price data
            rename: If True, rename columns from *_col parameters
            gross_col: Column name for gross charges (if rename=True)
            cash_col: Column name for cash prices (if rename=True)
            cpt_col: Column name for CPT codes (if rename=True)

        Returns:
            Normalized DataFrame
        """
        df = df.copy()

        # Rename columns if specified
        if rename and gross_col and cash_col and cpt_col:
            df["gross"] = df[gross_col]
            df["cash"] = df[cash_col]
            df["concept_code"] = df[cpt_col].apply(
                lambda x: self.strip_leading_zero(str(x).strip())
            )
            df["vocabulary_id"] = "cpt"

        # Ensure required columns exist
        required = ["vocabulary_id", "concept_code", "gross", "cash"]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        has_net = "net" in df.columns and df["net"].notna().any()

        # Clean CPT codes
        df["concept_code"] = df["concept_code"].apply(
            lambda x: self.strip_leading_zero(str(x).strip()) if pd.notna(x) else ""
        )

        # Clean price columns and ensure numeric dtype
        df["gross"] = df["gross"].apply(self.clean_price)
        df["cash"] = df["cash"].apply(self.clean_price)
        df["gross"] = pd.to_numeric(df["gross"], errors="coerce")
        df["cash"] = pd.to_numeric(df["cash"], errors="coerce")

        if has_net:
            df["net"] = df["net"].apply(self.clean_price)
            df["net"] = pd.to_numeric(df["net"], errors="coerce")
            if "payer" not in df.columns:
                df["payer"] = None
            if "plan" not in df.columns:
                df["plan"] = None

        df = self._filter_vocab(df)
        if df.empty:
            cols = ["cpt", "type", "price"]
            if has_net:
                cols.extend(["payer", "plan"])
            return pd.DataFrame(columns=cols)

        # --- Overall cash / gross (one max per CPT) ---
        overall = (
            df.groupby(["vocabulary_id", "concept_code"])[["cash", "gross"]]
            .max()
            .reset_index()
        )
        overall = pd.melt(
            overall,
            id_vars="concept_code",
            value_vars=["cash", "gross"],
            var_name="type",
            value_name="price",
        )
        overall = overall.rename(columns={"concept_code": "cpt"})
        overall["payer"] = pd.NA
        overall["plan"] = pd.NA

        frames = [overall]

        # --- Net by payer + plan ---
        if has_net:
            net_df = df[df["net"].notna() & df["payer"].notna() & df["plan"].notna()].copy()
            if not net_df.empty:
                net_agg = (
                    net_df.groupby(["concept_code", "payer", "plan"])["net"]
                    .max()
                    .reset_index()
                )
                net_agg = net_agg.rename(columns={"concept_code": "cpt", "net": "price"})
                net_agg["type"] = "net"
                frames.append(net_agg[["cpt", "type", "price", "payer", "plan"]])

        result = pd.concat(frames, ignore_index=True)

        # If no net rows survived, drop empty payer/plan columns for backward compat
        if not has_net or result["type"].eq("net").sum() == 0:
            result = result.drop(columns=["payer", "plan"], errors="ignore")

        return self._finalize(result)
