# image_reviewer.py
from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

# A simple default rubric (pipeline imports this symbol)
DEFAULT_RUBRIC = {
    "identity_consistency": "Characters match the provided reference sheets.",
    "line_art_quality": "Clean black line art only. No shading, no grayscale, no color.",
    "composition": "Clear focal point, good white space, printable coloring-book layout.",
    "anatomy": "No weird hands/limbs, no melted faces, no extra fingers.",
    "style_consistency": "Same line style across pages; not chibi; not painterly.",
}


@dataclass
class ReviewResult:
    best_index: int
    scores: list[float]
    reasons: list[str]
    needs_regen: bool
    edit_suggestion: str = ""


class ImageReviewer:
    """
    Minimal reviewer that keeps your pipeline stable.

    Modes (STORYBOOK_REVIEWER):
      - "off" (default): always accept candidate 0, no regen.
      - "basic": tiny heuristic based on filename (still basically accepts 0).
    """

    def __init__(self, mode: str = "off"):
        self.mode = (mode or "off").strip().lower()

    @classmethod
    def from_env(cls) -> ImageReviewer:
        return cls(mode=os.getenv("STORYBOOK_REVIEWER", "off"))

    def review(
        self,
        *,
        candidates: Sequence[Path],
        page_prompt: str,
        rubric: dict,
        page_title: str | None = None,
    ) -> ReviewResult:
        if not candidates:
            return ReviewResult(
                best_index=0,
                scores=[0.0],
                reasons=["No candidates provided."],
                needs_regen=True,
                edit_suggestion="Generate at least 1 candidate image.",
            )

        # Reviewer OFF: always pick the first and never regen.
        if self.mode == "off":
            return ReviewResult(
                best_index=0,
                scores=[10.0 for _ in candidates],
                reasons=["Reviewer off: accepting candidate 0." for _ in candidates],
                needs_regen=False,
                edit_suggestion="",
            )

        # BASIC: still basically accept 0; placeholder for future smarter scoring.
        scores = [9.0 for _ in candidates]
        reasons = ["Basic review: default accept." for _ in candidates]
        return ReviewResult(
            best_index=0,
            scores=scores,
            reasons=reasons,
            needs_regen=False,
            edit_suggestion="",
        )
