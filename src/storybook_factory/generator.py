from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import yaml
from jinja2 import Template

DEFAULT_TRIM = {"w": 8.5, "h": 11.0}
DEFAULT_DPI = 300
INTERIOR_PX = {"w": 2550, "h": 3300}
COVER_PX = {"w": 2625, "h": 3375}


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text())


def _render_template(s: str, ctx: dict[str, Any]) -> str:
    return Template(s).render(**ctx)


def _describe_child(c: dict[str, Any]) -> str:
    name = c.get("name", "Child")
    age = c.get("age")
    appearance = c.get("appearance_desc") or c.get("appearance") or ""
    outfit = c.get("outfit_desc") or c.get("outfits") or ""
    bits: list[str] = [name]
    if age:
        bits.append(f"age {age}")
    if appearance:
        bits.append(appearance)
    if outfit:
        bits.append(f"wearing {outfit}")
    return ", ".join(bits)


def _describe_pet(p: dict[str, Any]) -> str:
    name = p.get("name", "Pet")
    species = p.get("species")
    appearance = p.get("appearance_desc") or p.get("appearance") or ""
    bits: list[str] = [name]
    if species:
        bits.append(f"a {species}")
    if appearance:
        bits.append(appearance)
    return ", ".join(bits)


def _strip_leading_scene_label(text: str) -> str:
    # Helps if any YAML prompt still begins with "Scene:" — we don't need it.
    t = text.strip()
    if t.lower().startswith("scene:"):
        return t.split(":", 1)[1].strip()
    return t


def _build_context(brief: dict[str, Any], theme: dict[str, Any]) -> dict[str, Any]:
    children = brief.get("children", []) or []
    pets = brief.get("pets", []) or []

    child_names = [c.get("name", "Child") for c in children]
    pet_names = [p.get("name", "Pet") for p in pets]

    child_descs = [_describe_child(c) for c in children]
    pet_descs = [_describe_pet(p) for p in pets]

    def join_human(names: list[str]) -> str:
        if not names:
            return "the children"
        if len(names) == 1:
            return names[0]
        if len(names) == 2:
            return f"{names[0]} and {names[1]}"
        return ", ".join(names[:-1]) + f", and {names[-1]}"

    child_summary_sentence = (
        join_human(child_names) + " (" + "; ".join(d for d in child_descs) + ")"
        if child_descs
        else join_human(child_names)
    )

    if pet_descs:
        pet_summary_sentence = (
            join_human(pet_names) + " (" + "; ".join(d for d in pet_descs) + ")"
        )
    else:
        pet_summary_sentence = join_human(pet_names) if pet_names else "the pets"

    # Keep this SHORT. Over-specifying leads to dark/inverted outputs.
    line_art_style = (
        "Black-and-white kids' coloring book page. "
        "Simple, clean black outlines on a white background. "
        "No shading, no gray tones, no shadows, no gradients. "
        "No large filled black areas; background stays mostly white. "
        "All characters and objects are outlined only, leaving the insides white for coloring. "
        "Cute, friendly proportions with clear shapes and plenty of white space."
    )

    # Character bible (once per prompt)
    character_bible_lines: list[str] = []
    if child_descs:
        character_bible_lines.append(
            "Children (draw these the same way on every page):"
        )
        for desc in child_descs:
            character_bible_lines.append(f"- {desc}")
    else:
        character_bible_lines.append(
            "Children: use the same specific kids as described in the brief."
        )

    if pet_descs:
        character_bible_lines.append("Pets (draw these the same way on every page):")
        for desc in pet_descs:
            character_bible_lines.append(f"- {desc}")
    else:
        character_bible_lines.append(
            "Pets: use the same specific pets as described in the brief."
        )

    character_bible_lines.append(
        "Keep these characters consistent. Do NOT replace them with generic children or generic pets."
    )
    character_bible = "\n".join(character_bible_lines)

    ctx: dict[str, Any] = {
        "child_names": child_names,
        "pet_names": pet_names,
        "child_descriptions": child_descs,
        "pet_descriptions": pet_descs,
        "child_summary_sentence": child_summary_sentence,
        "pet_summary_sentence": pet_summary_sentence,
        "line_art_style": line_art_style,
        "character_bible": character_bible,
        # Back-compat if any templates still reference this:
        "character_consistency": character_bible,
        "mythic_elements": theme.get("mythic_elements", {}),
    }
    return ctx


def generate_from_brief(
    brief_path: Path, theme_path: Path, out_dir: Path
) -> dict[str, Any]:
    """
    Load a YAML brief + theme/scene-pack, construct a rich context (visual bible + cast),
    and emit:
      - visual_bible.json
      - page_prompts.json
      - pipeline_config.json
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    brief = load_yaml(brief_path)
    theme = load_yaml(theme_path)

    children = brief.get("children", []) or []
    pets = brief.get("pets", []) or []

    ctx = _build_context(brief, theme)

    visual_bible = {
        "visual_bible_version": "1.3",
        "project": brief.get("project_title", "Custom Storybook"),
        "page_format": {
            "trim_in": DEFAULT_TRIM,
            "bleed_in": 0.125,
            "dpi": DEFAULT_DPI,
            "pixels": INTERIOR_PX,
            "orientation": "portrait",
            "interior_color_space": "grayscale",
            "cover_color_space": "RGB-sRGB",
        },
        "style": {
            "medium": "coloring-book line art",
            "lines": {
                "primary_px": 5,
                "secondary_px": 3,
                "rules": [
                    "bold, clean outlines",
                    "no shading",
                    "no halftones",
                    "no gradients",
                    "no filled black areas",
                    "white background",
                ],
            },
            "bg_rules": "readable backgrounds; ample white space for coloring",
            "do_not": [
                "extra humans not listed in the cast",
                "duplicate or invented pets",
                "on-image text/logos except painted title/subtitle on front cover if explicitly requested",
                "comic panels or split frames",
            ],
        },
        "cast": {
            "children": children,
            "pets": pets,
        },
        "print_safety": {
            "interior_bleed_in": 0.125,
            "cover_bleed_in": 0.125,
            "safety_margin_in": 0.5,
        },
    }

    scenes = brief.get("scenes") or theme.get("scenes", [])
    interior_prompts: list[dict[str, Any]] = []

    for sc in scenes:
        page = sc["page"]
        title = sc["title"]
        prompt_t = sc["prompt"]  # scene-only template from YAML
        fname = f"page_{page:02d}_{slug(title)}.png"

        scene_text = _render_template(prompt_t, ctx)
        scene_text = _strip_leading_scene_label(scene_text)

        # Final interior prompt = style + bible + scene
        final_prompt = (
            ctx["line_art_style"].strip()
            + "\n\nCharacter bible:\n"
            + ctx["character_bible"].strip()
            + "\n\n"
            + scene_text.strip()
        )

        interior_prompts.append(
            {"page": page, "title": title, "file": fname, "prompt": final_prompt}
        )

    covers_cfg = theme.get("covers", {})
    covers: dict[str, Any] = {}

    for key, cov in covers_cfg.items():
        scene_text = _render_template(cov.get("prompt", ""), ctx).strip()

        final_cover_prompt = (
            "You are drawing a full-color storybook cover illustration.\n"
            "\nCharacter bible (keep consistent with the interior cast):\n"
            + ctx["character_bible"].strip()
            + "\n\n"
            + scene_text
        )

        covers[key] = {
            "file": cov.get("file"),
            "prompt": final_cover_prompt,
            "style_reference_image": cov.get("style_reference_image"),  # <-- add this
        }

    page_prompts = {"interior_prompts": interior_prompts, "covers": covers}

    pipeline_config = {
        "trim_in": DEFAULT_TRIM,
        "dpi": DEFAULT_DPI,
        "interior_pixels": INTERIOR_PX,
        "cover_pixels": COVER_PX,
        "project_title": visual_bible["project"],
    }

    (out_dir / "visual_bible.json").write_text(json.dumps(visual_bible, indent=2))
    (out_dir / "page_prompts.json").write_text(json.dumps(page_prompts, indent=2))
    (out_dir / "pipeline_config.json").write_text(json.dumps(pipeline_config, indent=2))

    return {
        "visual_bible": visual_bible,
        "page_prompts": page_prompts,
        "pipeline_config": pipeline_config,
    }
