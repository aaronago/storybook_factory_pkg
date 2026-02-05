# image_reviewer.py
from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import OpenAI

DEFAULT_RUBRIC = {
    "must_have": [
        "Black-and-white line art only (no color).",
        "No shading, no gray tones, no gradients, no shadows.",
        "No large solid filled black areas.",
        "Characters must match the provided character bible (no generic substitutions).",
        "No extra animals or extra people not in the cast.",
        "Composition is clear and readable for a child's coloring page.",
    ],
    "should_have": [
        "Line hierarchy: thicker outer contours for main subjects, thinner for background.",
        "Off-center or asymmetrical composition (avoid perfectly centered, stock layouts).",
        "One clear focal point and generous white space for coloring.",
        "Kid-friendly shapes with uncluttered background.",
    ],
    "penalize": [
        "Cross-hatching or heavy texture that looks like shading.",
        "Busy backgrounds that reduce colorable white space.",
        "Random extra props that distract from the scene.",
    ],
}

DEFAULT_SYSTEM = """You are an art director and QA reviewer for children's coloring-book line art.
You will be shown several candidate images for the same page prompt.
Score each candidate against the rubric and select the best.

Rules:
- Be strict about: no shading/gray, no gradients, no heavy fill, no extra animals/people.
- Prefer images that look hand-drawn (not stock), with line-weight hierarchy and clear focal point.
- Return JSON only with the schema requested.
"""


@dataclass
class ReviewResult:
    best_index: int
    scores: list[float]
    reasons: list[str]
    needs_regen: bool
    edit_suggestion: str | None = None


@dataclass
class ImageReviewer:
    """
    Uses a vision-capable Responses API model to score and rank candidates.
    """

    client: OpenAI
    model: str = "gpt-4.1"
    system_prompt: str = DEFAULT_SYSTEM
    temperature: float = 0.1
    max_output_tokens: int = 900

    @classmethod
    def from_env(cls) -> ImageReviewer:
        client = OpenAI()
        model = os.getenv("STORYBOOK_REVIEW_MODEL", "gpt-4.1")
        return cls(client=client, model=model)

    def review(
        self,
        *,
        candidates: list[Path],
        page_prompt: str,
        rubric: dict[str, Any] | None = None,
        page_title: str | None = None,
    ) -> ReviewResult:
        rubric = rubric or DEFAULT_RUBRIC

        # Allow disabling reviewer to compare raw output quickly.
        if os.getenv("STORYBOOK_DISABLE_IMAGE_REVIEW", "").lower() in {
            "1",
            "true",
            "yes",
        }:
            return ReviewResult(
                best_index=0,
                scores=[1.0 for _ in candidates],
                reasons=["review disabled" for _ in candidates],
                needs_regen=False,
                edit_suggestion=None,
            )

        content_blocks = [
            {
                "type": "input_text",
                "text": self._build_user_text(page_prompt, rubric, page_title),
            }
        ]

        # Attach candidate images
        for p in candidates:
            content_blocks.append(
                {
                    "type": "input_image",
                    "image_url": self._as_data_url(p),
                }
            )

        resp = self.client.responses.create(
            model=self.model,
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
            input=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": content_blocks},
            ],
        )

        data = self._extract_json(resp)

        # Expected shape:
        # {
        #   "best_index": 0,
        #   "scores": [0..10],
        #   "reasons": ["...", "..."],
        #   "needs_regen": true/false,
        #   "edit_suggestion": "..." | null
        # }
        best_index = int(data.get("best_index", 0))
        scores = data.get("scores") or []
        reasons = data.get("reasons") or []
        needs_regen = bool(data.get("needs_regen", False))
        edit_suggestion = data.get("edit_suggestion")

        # Defensive defaults
        if not scores or len(scores) != len(candidates):
            scores = [0.0 for _ in candidates]
            scores[best_index] = 1.0
        if not reasons or len(reasons) != len(candidates):
            reasons = ["(no reason provided)" for _ in candidates]

        best_index = max(0, min(best_index, len(candidates) - 1))

        return ReviewResult(
            best_index=best_index,
            scores=[float(s) for s in scores],
            reasons=[str(r) for r in reasons],
            needs_regen=needs_regen,
            edit_suggestion=str(edit_suggestion) if edit_suggestion else None,
        )

    def _build_user_text(
        self, page_prompt: str, rubric: dict[str, Any], page_title: str | None
    ) -> str:
        title = f"Page title: {page_title}\n\n" if page_title else ""
        return (
            f"{title}"
            "You are reviewing candidate images for a single coloring-book page.\n\n"
            "PAGE PROMPT (for reference):\n"
            f"{page_prompt}\n\n"
            "RUBRIC:\n"
            f"- MUST HAVE:\n  - " + "\n  - ".join(rubric.get("must_have", [])) + "\n\n"
            "- SHOULD HAVE:\n  - "
            + "\n  - ".join(rubric.get("should_have", []))
            + "\n\n"
            "- PENALIZE:\n  - " + "\n  - ".join(rubric.get("penalize", [])) + "\n\n"
            "Return JSON with:\n"
            '{ "best_index": int, "scores": number[], "reasons": string[], "needs_regen": bool, "edit_suggestion": string|null }\n'
            "Scores should be 0-10. needs_regen should be true if ALL candidates violate must-have constraints.\n"
        )

    def _as_data_url(self, path: Path) -> str:
        b = path.read_bytes()
        b64 = base64.b64encode(b).decode("ascii")
        return f"data:image/png;base64,{b64}"

    def _extract_json(self, resp: Any) -> dict[str, Any]:
        text = ""
        if hasattr(resp, "output_text"):
            text = resp.output_text or ""
        text = text.strip()

        import json
        import re

        if text.startswith("{") and text.endswith("}"):
            try:
                return json.loads(text)
            except Exception:
                pass

        chunks = []
        if hasattr(resp, "output") and resp.output:
            for item in resp.output:
                content = getattr(item, "content", None)
                if not content:
                    continue
                for c in content:
                    t = getattr(c, "text", None)
                    if t:
                        chunks.append(t)
        joined = "\n".join(chunks).strip()
        if not joined:
            joined = text

        m = re.search(r"\{.*\}", joined, flags=re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass

        return {}
