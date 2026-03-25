from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, TypedDict, cast

import yaml
from jinja2 import Template

# -----------------------------
# Defaults
# -----------------------------

DEFAULT_TRIM = {"w": 8.5, "h": 11.0}
DEFAULT_DPI = 300

# 8.5x11 @ 300dpi
INTERIOR_PX = {"w": 2550, "h": 3300}

# cover template may vary; keep as-is
COVER_PX = {"w": 2588, "h": 3375}

VISUAL_BIBLE_VERSION = "1.5"


# -----------------------------
# Utilities
# -----------------------------


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping at {path}, got: {type(data)}")
    return cast(dict[str, Any], data)


def _render_template(s: str, ctx: dict[str, Any]) -> str:
    return Template(s).render(**ctx)


def _render_template_multipass(s: str, ctx: dict[str, Any], *, passes: int = 3) -> str:
    """
    Render a template up to N times to allow globals that themselves contain templates.
    """
    rendered = str(s)
    for _ in range(passes):
        new_rendered = _render_template(rendered, ctx)
        if new_rendered == rendered:
            break
        rendered = new_rendered
    return rendered


def _strip_leading_scene_label(text: str) -> str:
    t = text.strip()
    if t.lower().startswith("scene:"):
        return t.split(":", 1)[1].strip()
    return t


def _describe_child(c: dict[str, Any]) -> str:
    name = c.get("name", "Child")
    age = c.get("age")
    appearance = c.get("appearance_desc") or c.get("appearance") or ""
    bits: list[str] = [str(name)]
    if age:
        bits.append(f"age {age}")
    if appearance:
        bits.append(str(appearance))
    return ", ".join(bits)


def _describe_pet(p: dict[str, Any]) -> str:
    name = p.get("name", "Pet")
    species = p.get("species")
    appearance = p.get("appearance_desc") or p.get("appearance") or ""
    bits: list[str] = [str(name)]
    if species:
        bits.append(f"a {species}")
    if appearance:
        bits.append(str(appearance))
    return ", ".join(bits)


def _join_human(names: list[str], fallback: str) -> str:
    if not names:
        return fallback
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return ", ".join(names[:-1]) + f", and {names[-1]}"


def _ensure_list(x: Any) -> list[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    raise ValueError(f"Expected list, got {type(x)}: {x!r}")


def _render_overlay_fields(ov2: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    """
    Render common overlay text fields. Supports both 'text' (current) and 'content' (legacy).
    """
    for field in ("text", "content", "label"):
        if isinstance(ov2.get(field), str):
            ov2[field] = _render_template_multipass(str(ov2[field]), ctx).strip()
    return ov2


def _render_overlays(
    overlays: Any, ctx: dict[str, Any], *, label: str
) -> list[dict[str, Any]]:
    if overlays is None:
        return []
    if not isinstance(overlays, list):
        raise ValueError(f"{label} must be a list. Got: {type(overlays)}")

    rendered_overlays: list[dict[str, Any]] = []
    for ov in overlays:
        if not isinstance(ov, dict):
            raise ValueError(f"{label} overlay must be mapping. Got: {type(ov)}")
        ov2 = dict(ov)
        rendered_overlays.append(_render_overlay_fields(ov2, ctx))
    return rendered_overlays


# -----------------------------
# Prompt item schema
# -----------------------------


class Pixels(TypedDict):
    w: int
    h: int


class PromptOut(TypedDict, total=False):
    page: int
    title: str
    file: str
    prompt: str
    pixels: Pixels
    kind: str
    overlays: list[dict[str, Any]]


@dataclass(frozen=True)
class PromptBuildDefaults:
    prefix: str
    pixels: Pixels
    include_bible: bool = True
    include_safety: bool = False


def _validate_pixels(p: Any, default_px: Pixels) -> Pixels:
    if p is None:
        return default_px
    if not isinstance(p, dict):
        raise ValueError(f"pixels must be a mapping like {{w,h}}. Got: {p!r}")
    w = p.get("w")
    h = p.get("h")
    if not isinstance(w, int) or not isinstance(h, int):
        raise ValueError(f"pixels must contain integer w/h. Got: {p!r}")
    return {"w": w, "h": h}


def _filename_for_item(
    title: str,
    *,
    page: int | None,
    prefix: str,
    explicit_file: str | None,
) -> str:
    if explicit_file:
        return explicit_file
    safe_title = slug(title) or "untitled"
    if page is None:
        return f"{prefix}_{safe_title}.png"
    return f"{prefix}_{int(page):02d}_{safe_title}.png"


def _build_prompt_item(
    item: dict[str, Any],
    ctx: dict[str, Any],
    defaults: PromptBuildDefaults,
    *,
    kind: str,
) -> PromptOut:
    title = item.get("title")

    # Select the right prompt variant based on cast size.
    # single_child_prompt is used when the brief has exactly 1 child.
    # no_pet_prompt is used when the brief has no pets.
    # Fall back to the standard prompt if the alternate isn't defined.
    child_count = len(ctx.get("child_names", []))
    pet_count = len(ctx.get("pet_names", []))

    if child_count <= 1 and item.get("single_child_prompt"):
        prompt_t = item.get("single_child_prompt")
    elif pet_count == 0 and item.get("no_pet_prompt"):
        prompt_t = item.get("no_pet_prompt")
    else:
        prompt_t = item.get("prompt")

    if not isinstance(title, str) or not title.strip():
        raise ValueError(f"{kind} item missing non-empty 'title': {item}")
    if not isinstance(prompt_t, str) or not prompt_t.strip():
        raise ValueError(f"{kind} item missing non-empty 'prompt': {item}")

    page_val = item.get("page")
    page: int | None
    if page_val is None:
        page = None
    else:
        if not isinstance(page_val, int):
            raise ValueError(f"{kind} item 'page' must be int. Got: {page_val!r}")
        page = page_val

    prefix = str(item.get("prefix") or defaults.prefix)
    pixels = _validate_pixels(item.get("pixels"), defaults.pixels)

    include_safety = bool(item.get("include_safety", defaults.include_safety))

    # Render the scene template (which can reference {{ globals.* }}) with multi-pass support
    rendered = _render_template_multipass(str(prompt_t), ctx, passes=3)
    rendered = _strip_leading_scene_label(rendered).strip()

    # Support for explicit prompt blocks (REFERENCES, TASK, STYLE, COMPOSITION, SCENE)
    ref_desc = str(ctx.get("ref_description_string") or "").strip()

    parts: list[str] = []

    if ref_desc and kind != "front_matter":
        parts.append(ref_desc)

    if kind != "front_matter":
        character_bible = ctx.get("character_bible", "").strip()
        if character_bible:
            # Use the full "CHARACTER BIBLE:" header only when there are actual
            # character description lines (text-appearance build).  For reference
            # builds the bible is just an outfit hint — no header needed.
            has_char_lines = character_bible.startswith(
                "Children"
            ) or character_bible.startswith("Pets")
            if has_char_lines:
                parts.append("CHARACTER BIBLE:\n" + character_bible)
            else:
                parts.append(character_bible)

    parts.append(rendered)

    if include_safety:
        global_safety = str(ctx.get("global_safety_rules") or "").strip()
        if global_safety:
            parts.append("GLOBAL SAFETY (IMPORTANT):\n" + global_safety)

    final_prompt = "\n\n".join(p for p in parts if p).strip()

    fname = _filename_for_item(
        title.strip(),
        page=page,
        prefix=prefix,
        explicit_file=cast(Optional[str], item.get("file")),
    )

    out: PromptOut = {
        "title": title.strip(),
        "file": fname,
        "prompt": final_prompt,
        "pixels": pixels,
        "kind": kind,
    }

    if page is not None:
        out["page"] = page

    # -----------------------------
    # Pass-through overlays (YAML -> JSON) + TEMPLATE RENDER
    # -----------------------------
    overlays = item.get("overlays")
    if overlays is not None:
        out["overlays"] = _render_overlays(overlays, ctx, label=f"{kind}.overlays")

    return out


# -----------------------------
# Context / bible (LEAN)
# -----------------------------


def _merge_globals(brief: dict[str, Any], theme: dict[str, Any]) -> dict[str, Any]:
    theme_globals = theme.get("globals", {}) or {}
    brief_globals = brief.get("globals", {}) or {}

    if not isinstance(theme_globals, dict):
        raise ValueError("theme.globals must be a mapping")
    if not isinstance(brief_globals, dict):
        raise ValueError("brief.globals must be a mapping")

    # brief overrides theme
    return {**theme_globals, **brief_globals}


def _build_context(brief: dict[str, Any], theme: dict[str, Any]) -> dict[str, Any]:
    children = cast(list[dict[str, Any]], brief.get("children", []) or [])
    pets = cast(list[dict[str, Any]], brief.get("pets", []) or [])

    child_names = [str(c.get("name", "Child")) for c in children]
    pet_names = [str(p.get("name", "Pet")) for p in pets]

    child_descs = [_describe_child(c) for c in children]
    pet_descs = [_describe_pet(p) for p in pets]

    # If reference images are present, suppress text appearance descriptions —
    # the photos are authoritative and text descriptions may conflict with them.
    has_refs = bool(brief.get("ref_description_string", ""))

    # -------------------------------------------------
    # Character bible (injected by pipeline unless disabled)
    # -------------------------------------------------
    # When reference sheets are attached, the images are the authoritative
    # character source — text descriptions would only conflict.  Emit only
    # the outfit_hint (if provided) so the model knows how to dress them.
    outfit_hint = str(brief.get("outfit_hint", "")).strip()

    if has_refs:
        # Reference build: no text appearance lines; outfit hint only if set
        character_bible = f"Outfit: {outfit_hint}" if outfit_hint else ""
    else:
        character_bible_lines: list[str] = []

        if child_descs:
            character_bible_lines.append(
                "Children (draw these the same way on every page):"
            )
            for desc in child_descs:
                character_bible_lines.append(f"- {desc}")

        if pet_descs:
            character_bible_lines.append(
                "Pets (draw these the same way on every page):"
            )
            for desc in pet_descs:
                character_bible_lines.append(f"- {desc}")
        else:
            character_bible_lines.append(
                "Pets: use the same specific pets as described in the brief."
            )

        character_bible = "\n".join(character_bible_lines)

    # -------------------------------------------------
    # Globals (YAML-owned reusable prompt fragments & settings)
    # brief.globals overrides theme.globals
    # -------------------------------------------------
    merged_globals = _merge_globals(brief, theme)

    # Normalize overlay styles so downstream is predictable
    overlay_styles = merged_globals.get("overlay_styles", {})
    if overlay_styles is None:
        overlay_styles = {}
    if not isinstance(overlay_styles, dict):
        raise ValueError("globals.overlay_styles must be a mapping (dict) if provided")

    # Normalize global safety rules
    global_safety_rules = merged_globals.get("global_safety_rules", "")
    if global_safety_rules is None:
        global_safety_rules = ""

    # Optional: a convenient “cast summary” string for templates
    cast_summary_parts: list[str] = []
    if child_names:
        cast_summary_parts.append("Children: " + ", ".join(child_names))
    if pet_names:
        cast_summary_parts.append("Pets: " + ", ".join(pet_names))
    cast_summary = " | ".join(cast_summary_parts) if cast_summary_parts else ""

    # -------------------------------------------------
    # Safe cast accessors for YAML templates.
    # Use these instead of child_names[0] / child_names[1] / pet_names[0]
    # so the pack degrades gracefully when fewer characters are provided.
    #
    #   child_1  — first child name, or "the child" if none
    #   child_2  — second child name; falls back to child_1 if only 1 child
    #   pet_1    — first pet name, or "their pet" if no pets
    # -------------------------------------------------
    child_1 = child_names[0] if len(child_names) >= 1 else "the child"
    child_2 = child_names[1] if len(child_names) >= 2 else child_1
    pet_1 = pet_names[0] if len(pet_names) >= 1 else "their pet"

    ctx: dict[str, Any] = {
        # raw lists (keep for join/loop usage)
        "child_names": child_names,
        "pet_names": pet_names,
        "children": children,
        "pets": pets,
        "cast_summary": cast_summary,
        # safe scalar accessors — always a non-empty string, never an index error
        "child_1": child_1,
        "child_2": child_2,
        "pet_1": pet_1,
        # reusable prompt fragments live here (YAML "globals")
        "globals": merged_globals,
        # convenience: styles for overlays (pipeline reads from JSON, but templates may use too)
        "overlay_styles": overlay_styles,
        # injected blocks
        "character_bible": character_bible,
        "character_names": child_names + pet_names,
        "global_safety_rules": global_safety_rules,
        "ref_description_string": brief.get("ref_description_string", ""),
        # optional
        "mythic_elements": theme.get("mythic_elements", {}),
    }
    return ctx


# -----------------------------
# Selection logic (brief overrides theme)
# -----------------------------


def _select_items(
    brief: dict[str, Any],
    theme: dict[str, Any],
    key: str,
) -> list[dict[str, Any]]:
    if key in brief:
        return cast(list[dict[str, Any]], _ensure_list(brief.get(key)))
    return cast(list[dict[str, Any]], _ensure_list(theme.get(key)))


def _sort_by_page_then_title(items: Iterable[PromptOut]) -> list[PromptOut]:
    def k(it: PromptOut) -> tuple[int, str]:
        p = it.get("page")
        page_int = int(p) if isinstance(p, int) else 10_000
        return (page_int, it.get("title", ""))

    return sorted(list(items), key=k)


# -----------------------------
# Main generator
# -----------------------------


def generate_from_brief(
    brief_path: Path,
    theme_path: Path,
    out_dir: Path,
    ref_description_string: str = "",
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    brief = load_yaml(brief_path)
    theme = load_yaml(theme_path)

    # Inject ref string into brief if provided via arg (e.g. from CLI build)
    if ref_description_string:
        brief["ref_description_string"] = ref_description_string

    children = cast(list[dict[str, Any]], brief.get("children", []) or [])
    pets = cast(list[dict[str, Any]], brief.get("pets", []) or [])

    # Collect character names for the pipeline to use later
    character_names = []
    for c in children:
        if c.get("name"):
            character_names.append(c["name"])
    for p in pets:
        if p.get("name"):
            character_names.append(p["name"])

    ctx = _build_context(brief, theme)

    visual_bible = {
        "visual_bible_version": VISUAL_BIBLE_VERSION,
        "project": brief.get("project_title", "Custom Storybook"),
        "character_names": character_names,
        "ref_description_string": ref_description_string,
        "page_format": {
            "trim_in": DEFAULT_TRIM,
            "bleed_in": 0.125,
            "dpi": DEFAULT_DPI,
            "pixels": INTERIOR_PX,
            "orientation": "portrait",
        },
        "cast": {"children": children, "pets": pets},
        "print_safety": {
            "interior_bleed_in": 0.125,
            "cover_bleed_in": 0.125,
            "safety_margin_in": 0.5,
        },
        "globals_keys": sorted(list((ctx.get("globals") or {}).keys())),
    }

    front_matter_items = _select_items(brief, theme, "front_matter")
    scene_items = _select_items(brief, theme, "scenes")
    back_matter_items = _select_items(brief, theme, "back_matter")

    front_defaults = PromptBuildDefaults(prefix="front_matter", pixels=INTERIOR_PX)
    interior_defaults = PromptBuildDefaults(
        prefix="page", pixels=INTERIOR_PX, include_safety=True
    )
    back_defaults = PromptBuildDefaults(prefix="back_matter", pixels=INTERIOR_PX)

    front_matter_prompts: list[PromptOut] = []
    for fm in front_matter_items:
        if not isinstance(fm, dict):
            raise ValueError(f"front_matter item must be mapping, got: {type(fm)}")
        front_matter_prompts.append(
            _build_prompt_item(fm, ctx, front_defaults, kind="front_matter")
        )

    interior_prompts: list[PromptOut] = []
    for sc in scene_items:
        if not isinstance(sc, dict):
            raise ValueError(f"scene item must be mapping, got: {type(sc)}")
        if "page" not in sc:
            raise ValueError(f"scene is missing 'page': {sc}")
        interior_prompts.append(
            _build_prompt_item(sc, ctx, interior_defaults, kind="scene")
        )

    back_matter_prompts: list[PromptOut] = []
    for bm in back_matter_items:
        if not isinstance(bm, dict):
            raise ValueError(f"back_matter item must be mapping, got: {type(bm)}")
        back_matter_prompts.append(
            _build_prompt_item(bm, ctx, back_defaults, kind="back_matter")
        )

    front_matter_prompts = _sort_by_page_then_title(front_matter_prompts)
    interior_prompts = _sort_by_page_then_title(interior_prompts)
    back_matter_prompts = _sort_by_page_then_title(back_matter_prompts)

    pages = [p["page"] for p in interior_prompts if "page" in p]
    if len(pages) != len(set(pages)):
        dupes = sorted({p for p in pages if pages.count(p) > 1})
        raise ValueError(f"Duplicate interior page numbers found: {dupes}")

    # Covers (optional)
    covers_cfg = theme.get("covers", {}) or {}
    if not isinstance(covers_cfg, dict):
        raise ValueError("theme.covers must be a mapping")

    covers: dict[str, Any] = {}
    for key, cov in covers_cfg.items():
        if not isinstance(cov, dict):
            raise ValueError(f"Cover entry must be mapping. key={key}, got={type(cov)}")

        # Multi-pass template render so {{ globals.* }} can themselves contain templates.
        scene_text = _render_template_multipass(
            str(cov.get("prompt", "")), ctx, passes=3
        ).strip()

        final_cover_prompt = (
            "You are drawing a full-color storybook cover illustration.\n"
            "Full color, rich saturation, painterly lighting. NOT black-and-white. NOT line art.\n"
            "\nCharacter bible (keep consistent with the interior cast):\n"
            + str(ctx["character_bible"]).strip()
            + "\n\n"
            + scene_text
        )

        # Generate filename from prefix + key (e.g., "cover_front.png")
        prefix = cov.get("prefix", "cover")
        filename = f"{prefix}_{key}.png"

        # Render overlays for covers (THIS is what fixes literal {{ ... }} on cover text)
        rendered_cover_overlays = _render_overlays(
            cov.get("overlays", []),
            ctx,
            label=f"covers.{key}.overlays",
        )

        covers[str(key)] = {
            "file": filename,
            "prompt": final_cover_prompt,
            "pixels": _validate_pixels(cov.get("pixels"), COVER_PX),
            "style_reference_image": cov.get("style_reference_image"),
            "title": cov.get("title"),
            "overlays": rendered_cover_overlays,
        }

    page_prompts = {
        "front_matter_prompts": front_matter_prompts,
        "interior_prompts": interior_prompts,
        "back_matter_prompts": back_matter_prompts,
        "covers": covers,
        "character_names": character_names,
        "ref_description_string": ref_description_string,
    }
    page_prompts["overlay_styles"] = ctx.get("overlay_styles", {})

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
