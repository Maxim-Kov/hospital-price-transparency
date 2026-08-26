"""Normalize hospital payer/plan names to a canonical lookup.

Maps messy MRF strings (e.g. "KANCARE AETNA", "United Healthcare") onto
stable payer and plan ids defined in dim/payers.json.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from ..utils.logger import get_logger

logger = get_logger(__name__)

# Punctuation / separators collapsed for matching
_NORMALIZE_RE = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True)
class NormalizedPayerPlan:
    """Result of mapping a raw payer/plan pair."""

    payer: str
    plan: str
    matched: bool
    skipped: bool = False


class PayerPlanNormalizer:
    """Maps raw payer_name / plan_name strings to canonical ids."""

    def __init__(self, lookup: dict | None = None):
        """Initialize from a lookup dict or load dim/payers.json.

        Args:
            lookup: Optional pre-loaded payers.json content. If None, loads
                from the project dim/payers.json path.
        """
        if lookup is None:
            lookup = self._load_default_lookup()

        self._skip_payers = {self.normalize_key(x) for x in lookup.get("skip_payers", [])}
        self._combined_rules: list[dict] = lookup.get("combined_rules", [])

        # Alias -> canonical id, sorted by alias length descending for longest match
        self._payer_aliases: list[tuple[str, str]] = []
        for payer_id, meta in lookup.get("payers", {}).items():
            for alias in meta.get("aliases", []):
                key = self.normalize_key(alias)
                if key:
                    self._payer_aliases.append((key, payer_id))
            # Always include the id itself as an alias
            self._payer_aliases.append((self.normalize_key(payer_id), payer_id))

        self._plan_aliases: list[tuple[str, str]] = []
        for plan_id, meta in lookup.get("plans", {}).items():
            for alias in meta.get("aliases", []):
                key = self.normalize_key(alias)
                if key:
                    self._plan_aliases.append((key, plan_id))
            self._plan_aliases.append((self.normalize_key(plan_id), plan_id))

        # Longest alias first so "medicare advantage aetna" beats "aetna"
        self._payer_aliases.sort(key=lambda x: len(x[0]), reverse=True)
        self._plan_aliases.sort(key=lambda x: len(x[0]), reverse=True)

        self._unmapped: set[tuple[str, str]] = set()

    @staticmethod
    def _load_default_lookup() -> dict:
        """Load dim/payers.json relative to the project root."""
        path = Path(__file__).resolve().parents[2] / "dim" / "payers.json"
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    @classmethod
    def from_file(cls, path: Path) -> PayerPlanNormalizer:
        """Create a normalizer from an explicit JSON path."""
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f))

    @staticmethod
    def normalize_key(value: str | None) -> str:
        """Lowercase and strip punctuation for matching."""
        if value is None:
            return ""
        text = str(value).strip().lower()
        if not text:
            return ""
        return _NORMALIZE_RE.sub(" ", text).strip()

    @property
    def unmapped_pairs(self) -> set[tuple[str, str]]:
        """Raw (payer, plan) pairs that did not fully map."""
        return set(self._unmapped)

    def clear_unmapped(self) -> None:
        """Reset the unmapped log (e.g. between hospitals)."""
        self._unmapped.clear()

    def _match_alias(self, text: str, aliases: list[tuple[str, str]]) -> str | None:
        """Return canonical id if text equals or contains a known alias."""
        if not text:
            return None
        for alias, canonical in aliases:
            if text == alias or alias in text:
                return canonical
        return None

    def _apply_combined_rules(self, combined: str) -> tuple[str, str] | None:
        """Apply multi-token rules on the joined payer+plan string."""
        for rule in self._combined_rules:
            needles = [self.normalize_key(c) for c in rule.get("contains", [])]
            if needles and all(n in combined for n in needles if n):
                return rule["payer"], rule["plan"]
        return None

    def normalize(
        self,
        payer_raw: str | None,
        plan_raw: str | None = None,
    ) -> NormalizedPayerPlan:
        """Map raw payer/plan names to canonical ids.

        Returns skipped=True for chargemaster placeholders (CDM DEFAULT).
        Unmapped payers become payer='unmapped', plan='other' (or a matched plan).
        """
        payer_key = self.normalize_key(payer_raw)
        plan_key = self.normalize_key(plan_raw)

        if not payer_key or payer_key in self._skip_payers:
            return NormalizedPayerPlan(payer="unmapped", plan="other", matched=False, skipped=True)

        combined = f"{payer_key} {plan_key}".strip()

        # Combined rules first (handles "MEDICARE ADVANTAGE AETNA" style labels)
        combined_hit = self._apply_combined_rules(combined)
        if combined_hit:
            return NormalizedPayerPlan(
                payer=combined_hit[0],
                plan=combined_hit[1],
                matched=True,
            )

        payer = self._match_alias(payer_key, self._payer_aliases)
        if payer is None:
            payer = self._match_alias(combined, self._payer_aliases)

        plan = self._match_alias(plan_key, self._plan_aliases) if plan_key else None
        if plan is None:
            plan = self._match_alias(combined, self._plan_aliases)

        if plan is None:
            plan = "other"

        matched = payer is not None
        if not matched:
            payer = "unmapped"
            raw_pair = (str(payer_raw or "").strip(), str(plan_raw or "").strip())
            if raw_pair not in self._unmapped:
                self._unmapped.add(raw_pair)
                logger.info(
                    "unmapped_payer_plan",
                    payer_raw=raw_pair[0],
                    plan_raw=raw_pair[1],
                )

        return NormalizedPayerPlan(payer=payer, plan=plan, matched=matched)
