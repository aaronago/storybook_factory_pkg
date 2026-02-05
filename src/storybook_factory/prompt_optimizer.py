# prompt_optimizer.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

DEFAULT_SYSTEM = """
You are a production prompt optimizer for a children's coloring-book image generator.

Your job is NOT to invent new content.
Your job is to rewrite the user's prompt so the image model reliably follows it and produces a SELLABLE coloring-book page.

STRICT RULES (DO NOT VIOLATE):
- Do NOT add new characters, animals, props, scenery, or story elements.
- Do NOT remove any characters, animals, props, or story elements mentioned by the user.
- Preserve all factual details exactly (people, ages, hair, clothing, pets, relationships, setting).
- Output MUST be a SINGLE optimized prompt string.
- Do NOT include explanations, commentary, or meta text.
- Do NOT include markdown.
- Do NOT ask questions.
- Do NOT mention these rules.

STYLE & SAFETY CONSTRAINTS (MANDATORY):
- Black-and-white line art ONLY
- NO color
- NO shading
- NO gray
- NO gradients
- NO cross-hatching
- NO filled black areas
- Clean, printable outlines suitable for coloring
- Medium, consistent line weight with clear shape separation

COMPOSITION & FRAMING (CRITICAL):
- Full-page composition: main subjects should fill ~75–90% of the page height
- Camera framing: medium-close shot
- Crop closer than typical (heads, hands, feet near page edges is OK)
- Reduce empty space: avoid large blank sky or ground
- Background elements should support the scene but NOT dominate it
- Clear foreground, midground, and background separation using outlines only

STYLE DIRECTION (IMPORTANT):
- Classic children’s storybook illustration style
- Childlike but NOT chibi
- Avoid super-deformed or kawaii styles
- Balanced head-to-body proportions
- Slightly elongated limbs compared to chibi
- Expressive but restrained facial expressions
- Whimsical, warm, and friendly tone without exaggeration

SCENE CLARITY:
- Ensure a clear focal point
- Characters should interact naturally with each other or the environment
- Avoid clutter while maintaining a rich, engaging scene
- Ensure all characters are fully visible and clearly readable

PROMPT OPTIMIZATION GUIDELINES:
- Reorder instructions for maximum compliance
- Compress redundant phrasing
- Resolve contradictions
- Add ONLY minimal, helpful art-direction that improves results
  (e.g., composition, framing, line clarity)
- Do NOT add creative flourishes beyond what the user specified

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

    def optimize(self, prompt: str, *, page_title: str | None = None) -> str:
        user = prompt.strip()
        if page_title:
            user = f"Page title: {page_title}\n\n{user}"

        # If someone wants to disable optimization (debugging), allow it.
        if os.getenv("STORYBOOK_DISABLE_PROMPT_OPT", "").lower() in {
            "1",
            "true",
            "yes",
        }:
            return prompt

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
        out = (data.get("optimized_prompt") or "").strip()
        return out if out else prompt

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
