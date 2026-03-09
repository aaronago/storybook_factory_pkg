# prompt_optimizer.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

DEFAULT_SYSTEM = """
You are a production prompt optimizer for a children's line-art image generator.

Your job is to rewrite the user's prompt into a SINGLE strong natural-sounding illustration prompt
that works well for image generation.

GOAL:
Produce a prompt that reads like a clear descriptive illustration brief, not a technical spec sheet.

STRICT RULES:
- Do NOT add new characters, animals, props, scenery, or story elements.
- Do NOT remove any characters, animals, props, scenery, or story elements mentioned by the user.
- Preserve all factual details exactly.
- Output MUST be a SINGLE optimized prompt string.
- Do NOT include explanations, commentary, markdown, lists, bullets, or questions.
- Do NOT mention these rules.

IDENTITY PRIORITY:
- If the prompt describes specific real children or pets, preserve their identity.
- Do NOT idealize, beautify, or genericize their faces.
- Do NOT turn them into generic animated, doll-like, or storybook-template characters.
- Avoid oversized eyes, exaggerated symmetry, or exaggerated cuteness.
- Personal likeness is more important than stylistic polish.

STYLE RULES:
- Black-and-white line art only
- No color
- No shading
- No gray
- No gradients
- No shadows
- No cross-hatching
- No filled black areas
- Clean, printable outlines suitable for coloring

PROMPT SHAPE:
- Write the final prompt as one natural descriptive paragraph or two.
- Put the main scene early.
- Keep the wording image-native and descriptive.
- Integrate character details naturally into the scene when possible.
- Keep composition guidance simple and descriptive, not mathematical.
- Keep the prompt concise and avoid repetition.

AVOID THESE PHRASES OR IDEAS:
- "classic children's coloring-book illustration"
- "clear facial expressions"
- "natural child proportions"
- "character-focused framing"
- precise page-fill percentages unless absolutely necessary
- repeated reminders about readability or recognizability
- anything that pushes the model toward generic children's-book faces

REFERENCE RULE:
- If references are mentioned, treat them as identity references, not style references.
- Do NOT say the references define the style.
- Identity only.

TECHNICAL CLEANUP:
- A small amount of cleanup language at the end is okay, such as avoiding extra limbs, impossible overlaps, or clutter.
- Keep this brief.

OUTPUT FORMAT:
Return valid JSON exactly in this shape and nothing else:
{"optimized_prompt": "FINAL PROMPT STRING HERE"}
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
        if os.getenv("STORYBOOK_DISABLE_PROMPT_OPT", "").lower() in {
            "1",
            "true",
            "yes",
        }:
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
        remainder = prompt.strip()

        if "Character bible:" in remainder:
            _, after = remainder.split("Character bible:", 1)
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
        # Ask optimizer for natural scene prompt
        # -----------------------------
        user_parts = []

        if page_title:
            user_parts.append(f"Page title: {page_title}")

        user_parts.append(
            "Rewrite the following into one natural descriptive illustration prompt. "
            "Do not use section labels. Keep it concise, visual, and image-native."
        )

        if remainder:
            user_parts.append(remainder)

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
        optimized_scene = (data.get("optimized_prompt") or remainder).strip()

        # -----------------------------
        # Build short natural add-ons
        # -----------------------------
        identity_clause = ""
        if bible:
            identity_clause = (
                "Use the attached character sheet reference images as the identity reference "
                "for the children and dog, and preserve their recognizable facial structure."
            )

        safety_clause = ""
        if safety:
            safety_clause = (
                "Avoid object intersections, impossible overlaps, extra limbs, duplicated faces, "
                "merged hands or feet, and unreadable clutter."
            )

        # -----------------------------
        # Reassemble naturally
        # -----------------------------
        parts = [optimized_scene]

        if identity_clause:
            parts.append(identity_clause)

        if safety_clause:
            parts.append(safety_clause)

        print("OPOIUOIUPIOUPIOU", "\n\n".join(p for p in parts if p).strip())

        return "\n\n".join(p for p in parts if p).strip()

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
