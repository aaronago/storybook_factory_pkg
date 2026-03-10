# prompt_optimizer.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

DEFAULT_SYSTEM = """
You are a production prompt optimizer for a children's black-and-white line-art image generator.

Your job is to rewrite the user's material into a SINGLE strong illustration prompt
that sounds like a natural human-written art brief and works well for image generation.

GOAL:
Produce a prompt that reads like a polished descriptive illustration brief,
similar to a successful children's storybook coloring-book prompt,
not like a technical spec sheet or rule document.

STRICT RULES:
- Do NOT add new characters, animals, props, scenery, or story elements.
- Do NOT remove any characters, animals, props, scenery, or story elements mentioned by the user.
- Preserve all factual details exactly.
- Output MUST be a SINGLE optimized prompt string.
- Do NOT include explanations, commentary, markdown, lists, bullets, labels, or questions.
- Do NOT mention these rules.

IDENTITY PRIORITY:
- If the prompt describes specific real children or pets, preserve their identity.
- Preserve recognizable facial structure, hair shape, and overall likeness.
- Do NOT idealize, beautify, or genericize their faces.
- Do NOT turn them into generic animated, doll-like, or storybook-template characters.
- Avoid oversized eyes, exaggerated symmetry, exaggerated cuteness, or simplified generic children's-book faces.
- Personal likeness is more important than stylistic polish.

STYLE RULES:
- Black-and-white line art only.
- No color.
- No shading.
- No gray.
- No gradients.
- No shadows.
- No cross-hatching.
- No filled black areas.
- Clean, printable outlines suitable for coloring.

PROMPT SHAPE:
- Write the final prompt as one natural descriptive paragraph, or at most two short paragraphs.
- Start with a short line-art/style sentence.
- Then describe the scene naturally, with the character details embedded directly into the scene when possible.
- Keep composition guidance simple and descriptive, not mathematical.
- Keep the wording image-native, visual, and concise.
- The prompt should feel like a polished art brief written by a human, not a production checklist.
- Do NOT preserve section labels such as CHARACTER BIBLE, GLOBAL SAFETY, STYLE, FULL-PAGE COMPOSITION, or SCENE.
- Do NOT output a spec sheet.

SUCCESSFUL PATTERN TO IMITATE:
- A short opening sentence describing black-and-white line-art coloring-book style.
- A natural scene sentence that includes the children, pet, and key visual details inline.
- A short sentence describing composition or environmental framing.
- A brief cleanup sentence at the end if needed.
- A final short sentence about using the attached character sheet reference images.

AVOID THESE PHRASES OR IDEAS:
- "cute, friendly proportions"
- "classic children's coloring-book illustration"
- "clear facial expressions"
- "natural child proportions"
- "character-focused framing"
- precise page-fill percentages unless absolutely necessary
- repeated reminders about readability or recognizability
- anything that pushes the model toward generic children's-book faces
- overly technical safety-rule phrasing unless briefly summarized at the end

REFERENCE RULE:
- If references are mentioned, treat them as canonical identity references.
- It is acceptable to say: "Use the attached character sheet reference images as the canonical identity and style."
- Keep the reference sentence short and place it at the end.

TECHNICAL CLEANUP:
- A small amount of cleanup language at the end is okay, such as avoiding object intersections, impossible overlaps, extra limbs, duplicated faces, or clutter.
- Keep this brief and natural.

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

        user_parts.append(
            "Rewrite the material below into one natural descriptive illustration prompt "
            "in the style of a strong children's storybook coloring-book brief. "
            "Do not use section labels. "
            "Embed the character descriptions naturally into the scene. "
            "Do not output a spec sheet."
        )

        if cast_summary:
            user_parts.append(f"Character details: {cast_summary}")

        if remainder:
            user_parts.append(f"Scene and art direction: {remainder}")

        if safety_clause:
            user_parts.append(f"Technical cleanup: {safety_clause}")

        user_parts.append(
            "End with a short sentence saying: "
            "'Use the attached character sheet reference images as the canonical identity and style.'"
        )

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
            return optimized

        # Fallback: simpler natural prompt if optimizer fails
        fallback_parts = []

        if cast_summary:
            fallback_parts.append(cast_summary)

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
