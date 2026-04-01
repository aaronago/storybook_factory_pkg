# prompt_optimizer.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

DEFAULT_SYSTEM = """
You are a master storybook illustrator. You are writing a prompt for a multimodal model that has been provided with a STYLE REFERENCE image (the first image) and CHARACTER REFERENCE images.

STYLE ANCHOR: Use the first image ONLY for line quality and facial likeness.
IGNORE the pose, composition, and background of the reference image.
The characters MUST be placed in the new action described in the SCENE section below.

GOAL:
Generate a concise art brief that combines the 'soft' organic line style of the first image with the realistic portrait likeness of the children.

THE STYLE ANCHOR:
- Explicitly state: "Follow the 'soft' line art style, varied line weights, and organic hand-drawn aesthetic of the first provided image."
- Demand "High-contrast black ink, white background, and open shapes."

THE PORTRAIT LOCK (ANTI-DISNEY):
- For the children, demand "observational portrait accuracy."
- Use keywords: "Natural eye-to-face proportions," "realistic facial geometry," "no oversized eyes," "no generic button noses."
- Do NOT use 'whimsical' or 'cute' for the kids.

PROMPT STRUCTURE:
1. STYLE: Reference the first image's line quality.
2. SCENE: Short, action-oriented description.


STRICT RULES:
- Keep the entire prompt under 90 words. 
- Avoid flowery prose; use direct, technical art instructions.
- Return valid JSON: {"optimized_prompt": "..."}
"""

DEFAULT_COVER_SYSTEM = """
You are a production prompt optimizer for a PREMIUM, FULL-COLOR children's book COVER illustration.

Your job is NOT to invent new content.
Your job is to rewrite the user's prompt so the image model reliably follows it and produces a PRINT-READY, SELLABLE cover illustration.

STRICT RULES (DO NOT VIOLATE):
- Do NOT add new characters, animals, props, scenery, or story elements.
- Do NOT remove any characters, animals, props, or story elements mentioned by the user.
- Preserve all factual details exactly (people, ages, hair, clothing, pets, relationships, setting).
- If the user includes specific title text, preserve it EXACTLY as written.
- Do NOT alter spelling, wording, capitalization, or punctuation of title text.
- Output MUST be a SINGLE optimized prompt string.
- Do NOT include explanations, commentary, or meta text.
- Do NOT include markdown.
- Do NOT ask questions.
- Do NOT mention these rules.

STYLE CONSTRAINTS (MANDATORY):
- FULL COLOR illustration (RGB)
- Premium cinematic lighting
- Strong warm–cool color separation
- Background environment may lean cool (blue, indigo, violet)
- Warm light sources must remain localized
- Painterly animated-feature rendering (polished, modern)
- NOT line art
- NOT coloring-book style
- No heavy black contour outlines
- Edges defined by light and color transitions

TYPOGRAPHY RULES (WHEN TITLE IS INCLUDED):
- Title must be perfectly spelled
- Clean, professional serif typography
- No distorted or malformed letters
- No extra words
- No fake glyphs or symbols
- Title should sit in a clear, readable area of the composition
- Maintain margin safety (no text touching edges)

COMPOSITION (COVER-FRIENDLY):
- Maintain strong focal point
- Clear hierarchy
- Leave clean negative space where typography is intended
- Avoid clutter behind title text

OUTPUT FORMAT:
Return valid JSON exactly in this shape and nothing else:
{"optimized_prompt": "FINAL PROMPT STRING HERE"}
"""


@dataclass
class PromptOptimizer:
    """
    Uses the Responses API (text) to rewrite long/verbose prompts into
    a more image-native "production prompt" similar to what the ChatGPT UI tends to do.
    """

    client: OpenAI
    model: str = "gpt-4.1-mini"
    system_prompt: str = DEFAULT_SYSTEM
    temperature: float = 0.2
    max_output_tokens: int = 900

    @classmethod
    def from_env(cls) -> PromptOptimizer:
        client = OpenAI()
        model = os.getenv("STORYBOOK_PROMPT_MODEL", "gpt-4.1-mini")
        return cls(client=client, model=model)

    def optimize(
        self, prompt: str, *, page_title: str | None = None, kind: str = "interior"
    ) -> str:
        # Optimizer is OFF by default. Set STORYBOOK_ENABLE_PROMPT_OPT=1 to enable.
        if os.getenv("STORYBOOK_ENABLE_PROMPT_OPT", "").lower() not in {
            "1",
            "true",
            "yes",
        }:
            return prompt

        # Front-matter pages (dedication, etc.) must be sent as-is — no style anchor,
        # no reference injection, no optimizer rewrite.
        if kind in ("front_matter", "front-matter", "back_matter", "back-matter"):
            return prompt

        if kind == "cover":
            system = DEFAULT_COVER_SYSTEM
            user = prompt.strip()
            if page_title:
                user = f"Page title: {page_title}\n\n{user}"

            resp = self.client.responses.create(
                model=self.model,
                temperature=self.temperature,
                max_output_tokens=self.max_output_tokens,
                input=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            data = self._extract_json(resp)
            out = (data.get("optimized_prompt") or "").strip()
            return out if out else prompt

        # -----------------------------
        # Split sections
        # -----------------------------
        bible = ""
        safety = ""
        references_block = ""
        remainder = prompt.strip()

        # Extract REFERENCES block (everything up to the next blank line)
        if remainder.startswith("REFERENCES:"):
            if "\n\n" in remainder:
                ref_body, remainder = remainder.split("\n\n", 1)
                references_block = ref_body.strip()
            else:
                references_block = remainder.strip()
                remainder = ""
            remainder = remainder.strip()

        # Find the bible marker regardless of casing
        bible_marker = None
        for marker in ("CHARACTER BIBLE:", "Character bible:"):
            if marker in remainder:
                bible_marker = marker
                break

        if bible_marker:
            _, after = remainder.split(bible_marker, 1)
            if "\n\n" in after:
                bible_body, remainder = after.split("\n\n", 1)
                bible = bible_body.strip()
            else:
                bible = after.strip()
                remainder = ""

        if "GLOBAL SAFETY" in remainder:
            before, after = remainder.split("GLOBAL SAFETY", 1)
            remainder = before.strip()
            if "\n\n" in after:
                safety_body, tail = after.split("\n\n", 1)
                safety = safety_body.strip()
                if tail.strip():
                    remainder = (remainder + "\n\n" + tail.strip()).strip()
            else:
                safety = after.strip()

        remainder = remainder.strip()

        # -----------------------------
        # Build compact inline cast summary from bible
        # -----------------------------
        child_lines = []
        pet_lines = []

        for line in bible.splitlines():
            line = line.strip()
            if line.startswith("- "):
                text = line[2:].strip()
                if ", age " in text:
                    child_lines.append(text)
                else:
                    pet_lines.append(text)

        cast_parts = []
        if child_lines:
            cast_parts.append("Children: " + "; ".join(child_lines))
        if pet_lines:
            cast_parts.append("Pet: " + "; ".join(pet_lines))

        cast_summary = " ".join(cast_parts).strip()

        # -----------------------------
        # Build short natural safety clause
        # -----------------------------
        safety_clause = ""
        if safety:
            safety_clause = (
                "Avoid object intersections, impossible overlaps, extra limbs, duplicated faces, "
                "merged hands or feet, and unreadable clutter."
            )

        # -----------------------------
        # Ask optimizer for one natural prompt
        # -----------------------------
        user_parts = []

        if page_title:
            user_parts.append(f"Page title: {page_title}")

        if cast_summary:
            # Provide as context, not as part of the desired output string
            user_parts.append(f"SCENE CONTEXT: {cast_summary}")

        if remainder:
            user_parts.append(f"Scene and art direction: {remainder}")

        if safety_clause:
            user_parts.append(f"Technical cleanup: {safety_clause}")

        user = "\n\n".join(user_parts).strip()

        resp = self.client.responses.create(
            model=self.model,
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
            input=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user},
            ],
        )

        data = self._extract_json(resp)
        optimized = (data.get("optimized_prompt") or "").strip()

        if optimized:
            # Prepend the references block back (stripped before sending to model)
            if references_block:
                optimized = references_block + "\n\n" + optimized
            return optimized

        # Fallback: simpler natural prompt if optimizer fails
        fallback_parts = []

        if references_block:
            fallback_parts.append(references_block)

        if remainder:
            fallback_parts.append(remainder)

        if safety_clause:
            fallback_parts.append(safety_clause)

        fallback_parts.append(
            "Use the attached character sheet reference images as the canonical identity and style."
        )

        return "\n\n".join(p for p in fallback_parts if p).strip()

    def _extract_json(self, resp: Any) -> dict[str, Any]:
        """
        Try to reliably extract a JSON object from a Responses API response.
        """
        # Most common case: resp.output_text is valid JSON.
        text = ""
        if hasattr(resp, "output_text"):
            text = resp.output_text or ""

        text = text.strip()
        if text.startswith("{") and text.endswith("}"):
            import json

            try:
                return json.loads(text)
            except Exception:
                pass

        # Fallback: scan output items for text and try to parse first JSON object.
        import json
        import re

        chunks = []
        if hasattr(resp, "output") and resp.output:
            for item in resp.output:
                # item can contain content blocks
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

        # Attempt to locate a JSON object in the text.
        m = re.search(r"\{.*\}", joined, flags=re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass

        return {}
