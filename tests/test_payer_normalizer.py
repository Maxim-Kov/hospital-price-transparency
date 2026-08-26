"""Tests for payer/plan name normalization."""

import pytest

from src.normalizers.payer_normalizer import PayerPlanNormalizer


@pytest.fixture
def normalizer():
    """Load the default dim/payers.json lookup."""
    return PayerPlanNormalizer()


class TestPayerPlanNormalizer:
    """Tests for PayerPlanNormalizer using KS alias cases."""

    def test_aetna_commercial(self, normalizer):
        result = normalizer.normalize("AETNA", "COMMERCIAL")
        assert result.matched
        assert not result.skipped
        assert result.payer == "aetna"
        assert result.plan == "commercial"

    def test_medicare_advantage_aetna(self, normalizer):
        result = normalizer.normalize("MEDICARE ADVANTAGE AETNA", "MEDICARE ADVANTAGE AETNA")
        assert result.matched
        assert result.payer == "aetna"
        assert result.plan == "medicare_advantage"

    def test_aetna_medicare_advantage_reversed(self, normalizer):
        result = normalizer.normalize("AETNA MEDICARE ADVANTAGE", "AETNA MEDICARE ADVANTAGE")
        assert result.matched
        assert result.payer == "aetna"
        assert result.plan == "medicare_advantage"

    def test_kancare_aetna(self, normalizer):
        result = normalizer.normalize("KANCARE AETNA", "MEDICAID ADVANTAGE KANCARE AETNA")
        assert result.matched
        assert result.payer == "aetna"
        assert result.plan == "medicaid"

    def test_united_case_variants(self, normalizer):
        upper = normalizer.normalize("UNITED HEALTHCARE", "COMMERCIAL")
        mixed = normalizer.normalize("United Healthcare", "Commercial")
        assert upper.payer == "unitedhealthcare"
        assert mixed.payer == "unitedhealthcare"
        assert upper.plan == "commercial"
        assert mixed.plan == "commercial"

    def test_bcbs_blue_choice(self, normalizer):
        result = normalizer.normalize("BLUE CROSS BLUE SHIELD", "BLUE CHOICE")
        assert result.matched
        assert result.payer == "bcbs"
        assert result.plan == "commercial"

    def test_choicecare_humana(self, normalizer):
        result = normalizer.normalize(
            "MEDICARE ADVANTAGE CHOICECARE NETWORK",
            "MEDICARE ADVANTAGE CHOICECARE",
        )
        assert result.matched
        assert result.payer == "humana"
        assert result.plan == "medicare_advantage"

    def test_centene_brands(self, normalizer):
        sunflower = normalizer.normalize(
            "KANCARE SUNFLOWER",
            "MEDICAID ADVANTAGE KANCARE SUNFLOWER",
        )
        allwell = normalizer.normalize("MEDICARE ADVANTAGE ALLWELL", "MEDICARE ADVANTAGE ALLWELL")
        ambetter = normalizer.normalize(
            "MEDICARE ADVANTAGE AMBETTER",
            "MEDICARE ADVANTAGE AMBETTER",
        )
        assert sunflower.payer == "centene"
        assert sunflower.plan == "medicaid"
        assert allwell.payer == "centene"
        assert allwell.plan == "medicare_advantage"
        assert ambetter.payer == "centene"
        assert ambetter.plan == "exchange"

    def test_tricare_and_va(self, normalizer):
        tricare = normalizer.normalize("TRICARE", "TRICARE")
        va = normalizer.normalize("VA", "VETERANS AFFAIRS")
        assert tricare.payer == "tricare"
        assert tricare.plan == "tricare"
        assert va.payer == "va"
        assert va.plan == "va"

    def test_cdm_default_skipped(self, normalizer):
        result = normalizer.normalize("CDM DEFAULT", "CDM DEFAULT")
        assert result.skipped
        assert not result.matched

    def test_unmapped_logged(self, normalizer):
        result = normalizer.normalize("WPPA", "COMMERCIAL")
        assert not result.matched
        assert not result.skipped
        assert result.payer == "unmapped"
        assert result.plan == "commercial"
        assert ("WPPA", "COMMERCIAL") in normalizer.unmapped_pairs

    def test_normalize_key_strips_punctuation(self):
        assert PayerPlanNormalizer.normalize_key("United-Healthcare!") == "united healthcare"
        assert PayerPlanNormalizer.normalize_key("  AETNA  ") == "aetna"
