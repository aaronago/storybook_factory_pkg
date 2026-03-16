# pipeline.py
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from PIL import Image

from .covers import build_cover_pdf
from .image_provider import ImageProvider, ensure_dir
from .overlay_renderer import TextStyle, apply_overlays
from .prompt_optimizer import PromptOptimizer

# -----------------------------
# Config defaults (no pipeline_config.json required)
# -----------------------------


DEFAULT_PIPELINE_CFG: dict[str, Any] = {
    # Output quality
    "dpi": 300,
    # Single-sided coloring book behavior
    "one_sided": True,
    "blank_after_interior": True,
    "blank_after_front_matter": False,
    "blank_after_back_matter": False,
    "pad_to_multiple_of_4": True,
    # Overlays default ON to apply text overlays (dedication, titles, etc).
    "disable_overlays": False,
    # Fallback pixel size if not provided elsewhere
    # (Ideally your generator sets pixels per page in page_prompts.json, but this keeps you safe.)
    "interior_pixels": {"w": 2550, "h": 3300},
    "cover_pixels": {"w": 2588, "h": 3375},
}


def _load_optional_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _merge_cfg(defaults: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = dict(defaults)
    for k, v in overrides.items():
        # shallow merge is fine for this use-case
        merged[k] = v
    return merged


# -----------------------------
# IO helpers
# -----------------------------


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _as_list_of_dicts(x: Any, key_name: str) -> list[dict[str, Any]]:
    if x is None:
        return []
    if not isinstance(x, list):
        raise ValueError(f"page_prompts['{key_name}'] must be a list")
    out: list[dict[str, Any]] = []
    for item in x:
        if not isinstance(item, dict):
            raise ValueError(f"page_prompts['{key_name}'] contains non-dict: {item!r}")
        out.append(item)
    return out


def _sorted_interior_prompts(page_prompts: dict[str, Any]) -> list[dict[str, Any]]:
    interior = page_prompts.get("interior_prompts", [])
    if not isinstance(interior, list):
        raise ValueError("page_prompts['interior_prompts'] must be a list")
    for p in interior:
        if not isinstance(p, dict):
            raise ValueError(f"interior_prompts contains non-dict: {p!r}")
        if "page" not in p or "file" not in p or "prompt" not in p:
            raise ValueError(f"interior prompt missing required fields: {p}")
    return sorted(interior, key=lambda x: x["page"])


def _load_reference_sheets(images_dir: Path) -> list[Path]:
    refs_dir = images_dir / "refs"
    if not refs_dir.exists():
        return []

    style_dir = refs_dir / "style"
    characters_dir = refs_dir / "characters"

    style_refs: list[Path] = []
    character_refs: list[Path] = []

    if style_dir.exists():
        style_refs = sorted(p for p in style_dir.glob("*.png") if p.is_file())

    if characters_dir.exists():
        character_refs = sorted(p for p in characters_dir.glob("*.png") if p.is_file())

    # IMPORTANT: style first, then characters
    return style_refs + character_refs


# -----------------------------
# Overlay helpers
# -----------------------------


def _build_overlay_styles(page_prompts: dict[str, Any]) -> dict[str, TextStyle]:
    raw = page_prompts.get("overlay_styles", {}) or {}
    if not isinstance(raw, dict):
        raise ValueError("page_prompts['overlay_styles'] must be a dict if present")

    styles: dict[str, TextStyle] = {}
    valid_keys = {
        "font_path",
        "size",
        "fill",
        "stroke_width",
        "stroke_fill",
        "line_spacing",
        "glow_radius",
        "glow_color",
        "glow_alpha",
    }

    for k, v in raw.items():
        if not isinstance(v, dict):
            raise ValueError(f"overlay_styles['{k}'] must be a dict")
        filtered = {key: val for key, val in v.items() if key in valid_keys}
        styles[k] = TextStyle(**filtered)
    return styles


def _apply_item_overlays_if_any(
    *,
    images_dir: Path,
    item: dict[str, Any],
    overlay_styles: dict[str, TextStyle],
    overlays_enabled: bool,
    force_grayscale: bool = True,
) -> None:
    if not overlays_enabled:
        return

    overlays = item.get("overlays")
    if not overlays:
        return

    if not isinstance(overlays, list):
        raise ValueError(f"item.overlays must be a list. Got: {type(overlays)}")

    img_path = images_dir / item["file"]
    if not img_path.exists():
        raise FileNotFoundError(f"Cannot apply overlays; missing image: {img_path}")

    img = Image.open(img_path)
    out = apply_overlays(img, overlays=overlays, styles=overlay_styles)
    if force_grayscale:
        out = out.convert("L")
    out.save(img_path)


# -----------------------------
# PDF assembly (single-sided optimized)
# -----------------------------


def build_interior_pdf(
    images_dir: Path,
    page_prompts: dict[str, Any],
    pipeline_cfg: dict[str, Any],
    out_pdf: Path,
) -> Path:
    images: list[Image.Image] = []

    def _load_grayscale(path: Path) -> Image.Image:
        return Image.open(path).convert("L")

    def _blank_like(ref: Image.Image) -> Image.Image:
        return Image.new("L", ref.size, 255)

    front = _as_list_of_dicts(
        page_prompts.get("front_matter_prompts"), "front_matter_prompts"
    )
    interior = _sorted_interior_prompts(page_prompts)
    back = _as_list_of_dicts(
        page_prompts.get("back_matter_prompts"), "back_matter_prompts"
    )

    def _sort_optional_page(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if any(isinstance(it.get("page"), int) for it in items):
            return sorted(
                items,
                key=lambda it: (
                    it.get("page") if isinstance(it.get("page"), int) else 10_000
                ),
            )
        return items

    front = _sort_optional_page(front)
    back = _sort_optional_page(back)

    one_sided = bool(pipeline_cfg.get("one_sided", True))
    blank_after_interior = bool(pipeline_cfg.get("blank_after_interior", True))
    blank_after_front_matter = bool(pipeline_cfg.get("blank_after_front_matter", False))
    blank_after_back_matter = bool(pipeline_cfg.get("blank_after_back_matter", False))
    pad_to_multiple_of_4 = bool(pipeline_cfg.get("pad_to_multiple_of_4", True))

    blank_ref: Image.Image | None = None

    def _append(img: Image.Image) -> None:
        nonlocal blank_ref
        images.append(img)
        if blank_ref is None:
            blank_ref = img

    # Front matter
    for fm in front:
        fname = fm.get("file")
        if not fname:
            raise ValueError(f"Front matter item missing 'file': {fm}")
        p = images_dir / fname
        if not p.exists():
            raise FileNotFoundError(f"Missing front-matter image: {p}")
        _append(_load_grayscale(p))
        if one_sided and blank_after_front_matter:
            images.append(_blank_like(blank_ref))

    # Interior coloring pages
    for pmt in interior:
        fname = pmt["file"]
        page_num = pmt["page"]
        p = images_dir / fname
        if not p.exists():
            raise FileNotFoundError(f"Missing interior page {page_num} image: {p}")
        _append(_load_grayscale(p))
        if one_sided and blank_after_interior:
            images.append(_blank_like(blank_ref))

    # Back matter
    for bm in back:
        fname = bm.get("file")
        if not fname:
            raise ValueError(f"Back matter item missing 'file': {bm}")
        p = images_dir / fname
        if not p.exists():
            raise FileNotFoundError(f"Missing back-matter image: {p}")
        _append(_load_grayscale(p))
        if one_sided and blank_after_back_matter:
            images.append(_blank_like(blank_ref))

    if not images:
        raise RuntimeError("No images found; cannot build interior PDF.")

    if pad_to_multiple_of_4:
        if blank_ref is None:
            blank_ref = images[0]
        while (len(images) % 4) != 0:
            images.append(_blank_like(blank_ref))

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    first, *rest = images
    dpi = int(pipeline_cfg.get("dpi", 300))
    first.save(out_pdf, "PDF", resolution=dpi, save_all=True, append_images=rest)
    return out_pdf


# -----------------------------
# Generation helpers
# -----------------------------


def _ensure_prompt_images(
    *,
    prompts: list[dict[str, Any]],
    images_dir: Path,
    provider: ImageProvider,
    optimizer: PromptOptimizer | None,
    ref_sheets: list[Path],
    ref_description_string: str = "",
    overlay_styles: dict[str, TextStyle],
    overlays_enabled: bool,
    cand_n: int,
    image_provider_mode: str,
    keep_candidates: bool,
    kind_label: str,
) -> None:
    use_refs = kind_label not in {"front-matter", "back-matter"}

    for item in prompts:
        fname = item.get("file")
        raw_prompt = item.get("prompt", "")
        title = item.get("title") or kind_label
        page_num = item.get("page")

        if not fname:
            raise ValueError(f"{kind_label} item missing 'file': {item}")
        if not raw_prompt:
            raise ValueError(f"{kind_label} item missing 'prompt': {item}")

        final_path = images_dir / fname

        if final_path.exists():
            _apply_item_overlays_if_any(
                images_dir=images_dir,
                item=item,
                overlay_styles=overlay_styles,
                overlays_enabled=overlays_enabled,
                force_grayscale=True,
            )
            print(f"[pipeline] skipping existing {kind_label} image: {fname}")
            continue

        if image_provider_mode == "folder":
            got = provider._copy_from_assets(fname)
            if got is None:
                provider._placeholder(
                    fname, f"[MISSING FILE] {raw_prompt}", cover=False
                )

            _apply_item_overlays_if_any(
                images_dir=images_dir,
                item=item,
                overlay_styles=overlay_styles,
                overlays_enabled=overlays_enabled,
                force_grayscale=True,
            )
            continue

        if image_provider_mode != "gpt-image":
            provider._placeholder(fname, raw_prompt, cover=False)
            _apply_item_overlays_if_any(
                images_dir=images_dir,
                item=item,
                overlay_styles=overlay_styles,
                overlays_enabled=overlays_enabled,
                force_grayscale=True,
            )
            continue

        assert optimizer is not None

        page_title = title
        if isinstance(page_num, int):
            page_title = f"{title} (page {page_num})"

        optimized = optimizer.optimize(raw_prompt, page_title=page_title)

        # Checking if raw_prompt already contains REFERENCES to avoid double-injection
        if use_refs and ref_sheets and "REFERENCES:" not in raw_prompt:
            ref_msg = "\n\nUse the attached character sheet reference images as the canonical identity and style."
            if ref_description_string:
                ref_msg += f" {ref_description_string}"
            optimized += ref_msg

        print(
            f"[pipeline] generating candidates for {kind_label}: {fname} ({page_title})"
        )

        cands = provider.generate_candidates(
            base_filename=fname,
            prompt=optimized,
            cover=False,
            n=cand_n,
            reference_images=(ref_sheets if use_refs else []),
        )

        best_candidate = cands[0]
        provider.finalize_candidate(
            candidate_path=best_candidate, final_filename=fname, cover=False
        )

        _apply_item_overlays_if_any(
            images_dir=images_dir,
            item=item,
            overlay_styles=overlay_styles,
            overlays_enabled=overlays_enabled,
            force_grayscale=True,
        )

        if keep_candidates:
            report = {
                "kind": kind_label,
                "page": page_num,
                "title": title,
                "prompt": optimized,
                "selected": "first",
                "num_candidates": len(cands),
                "candidates": [c.name for c in cands],
            }
            (images_dir / f"{Path(fname).stem}__review.json").write_text(
                json.dumps(report, indent=2)
            )
        else:
            provider.cleanup_candidates(base_filename=fname)


# -----------------------------
# Main entry
# -----------------------------


def run_pipeline(
    config_dir: Path,
    output_dir: Path,
    assets_dir: Path,
    image_provider_mode: str = "mock",
    openai_model: str | None = None,
    interior_model: str | None = None,
    cover_model: str | None = None,
    dry_run: bool = False,
    image_quality: str = "standard",
    ref_description_string: str = "",
) -> dict[str, Any]:
    """
    Pipeline expects config_dir contains:
      - page_prompts.json

    Optional:
      - pipeline_config.json (if you add it later, it will be merged over defaults)
    """
    interior_model = interior_model or openai_model
    cover_model = cover_model or openai_model

    config_dir = config_dir.resolve()
    output_dir = output_dir.resolve()
    assets_dir = assets_dir.resolve()
    ensure_dir(output_dir)

    # RE-GENERATE page_prompts.json if ref_description_string is provided or changed
    # In 'gpt-image' mode, we often regenerate prompts on the fly during build
    # if the YAMLs are newer than the JSONs.

    images_dir = output_dir / "images"
    ensure_dir(images_dir)

    page_prompts = load_json(config_dir / "page_prompts.json")

    # If you add pipeline_config.json later, it overrides defaults.
    cfg_overrides = _load_optional_json(config_dir / "pipeline_config.json")
    pipeline_cfg = _merge_cfg(DEFAULT_PIPELINE_CFG, cfg_overrides)

    # Size defaults (used by ImageProvider if your generator didn’t set per-page pixels)
    ip = pipeline_cfg.get("interior_pixels", {"w": 2550, "h": 3300})
    interior_px = (int(ip["w"]), int(ip["h"]))

    cp = pipeline_cfg.get("cover_pixels", ip)
    cover_px = (int(cp["w"]), int(cp["h"]))

    provider = ImageProvider(
        out_dir=images_dir,
        interior_px=interior_px,
        cover_px=cover_px,
        mode=image_provider_mode,
        cover_model=cover_model,
        interior_model=interior_model,
        assets_dir=assets_dir,
        dry_run=dry_run,
        image_quality=image_quality,
    )

    optimizer: PromptOptimizer | None = None
    if image_provider_mode == "gpt-image":
        optimizer = PromptOptimizer.from_env()

    cand_n = min(int(os.getenv("STORYBOOK_CANDIDATES_N", "2")), 2)
    keep_candidates = os.getenv("STORYBOOK_KEEP_CANDIDATES", "").lower() in {
        "1",
        "true",
        "yes",
    }

    ref_description_string = ref_description_string or page_prompts.get(
        "ref_description_string", ""
    )

    ref_sheets: list[Path] = []
    # Create the character reference string once for the whole build
    # Load character names from the page_prompts (which come from the brief)
    char_names = page_prompts.get("character_names", [])

    if not char_names:
        print("[pipeline] Warning: No character_names found in page_prompts.json")
        # Fallback to subdirs if for some reason character_names isn't in JSON
        chars_root = assets_dir / "characters"
        if chars_root.exists():
            subdirs = [d.name for d in chars_root.iterdir() if d.is_dir()]
            if subdirs:
                char_names = [d.capitalize() for d in subdirs]

    if not char_names:
        print("[pipeline] Error: No character names could be found. Sorting will fail.")

    if not ref_description_string:
        print("[pipeline] ref_description_string is empty, initiating sort...")
        # Load the brief data to get character names
        chars_root = assets_dir / "characters"

        # Collect all images from assets/characters/
        image_extensions = (".png", ".jpg", ".jpeg", ".webp")
        all_refs = []
        if chars_root.exists():
            all_refs = sorted(
                [
                    p
                    for p in chars_root.rglob("*")
                    if p.suffix.lower() in image_extensions
                ]
            )

        print(f"[pipeline] Found {len(all_refs)} reference images in {chars_root}")

        if all_refs and char_names:
            from .image_sorter import GeminiImageSorter

            try:
                sorter = GeminiImageSorter()
                print(
                    f"[pipeline] Sorting {len(all_refs)} reference images for characters: {', '.join(char_names)}..."
                )
                res_string, res_sheets = sorter.get_reference_mapping(
                    all_refs, char_names
                )
                ref_description_string = res_string
                ref_sheets = res_sheets
                print(
                    f"[pipeline] Sort complete. Found {len(ref_sheets)} identified images."
                )
            except Exception as e:
                print(
                    f"[pipeline] Warning: Gemini sorter failed: {e}. No references attached."
                )
                ref_sheets = []
    else:
        print(
            f"[pipeline] ref_description_string already exists: {ref_description_string[:50]}..."
        )
        # If we already have a string (from JSON), we still need the actual Path objects in the right order!
        chars_root = assets_dir / "characters"

        image_extensions = (".png", ".jpg", ".jpeg", ".webp")
        all_refs = []
        if chars_root.exists():
            all_refs = sorted(
                [
                    p
                    for p in chars_root.rglob("*")
                    if p.suffix.lower() in image_extensions
                ]
            )

        if all_refs and char_names:
            from .image_sorter import GeminiImageSorter

            try:
                sorter = GeminiImageSorter()
                print(
                    f"[pipeline] Re-sorting {len(all_refs)} reference images to match existing indices for: {', '.join(char_names)}..."
                )
                # Use the sorter to get the correct path order matching the index-based prompt
                _, ref_sheets = sorter.get_reference_mapping(all_refs, char_names)
                print(
                    f"[pipeline] Re-sort complete. Found {len(ref_sheets)} identified images."
                )
            except Exception as e:
                print(f"[pipeline] Warning: Gemini sorter failed: {e}.")
                ref_sheets = []

    overlay_styles = _build_overlay_styles(page_prompts)
    overlays_enabled = not bool(pipeline_cfg.get("disable_overlays", True))

    front_list = _as_list_of_dicts(
        page_prompts.get("front_matter_prompts"), "front_matter_prompts"
    )
    interior_list = _sorted_interior_prompts(page_prompts)
    back_list = _as_list_of_dicts(
        page_prompts.get("back_matter_prompts"), "back_matter_prompts"
    )

    if front_list:
        _ensure_prompt_images(
            prompts=front_list,
            images_dir=images_dir,
            provider=provider,
            optimizer=optimizer,
            ref_sheets=ref_sheets,
            ref_description_string=ref_description_string,
            overlay_styles=overlay_styles,
            overlays_enabled=overlays_enabled,
            cand_n=cand_n,
            image_provider_mode=image_provider_mode,
            keep_candidates=keep_candidates,
            kind_label="front-matter",
        )

    _ensure_prompt_images(
        prompts=interior_list,
        images_dir=images_dir,
        provider=provider,
        optimizer=optimizer,
        ref_sheets=ref_sheets,
        ref_description_string=ref_description_string,
        overlay_styles=overlay_styles,
        overlays_enabled=overlays_enabled,
        cand_n=cand_n,
        image_provider_mode=image_provider_mode,
        keep_candidates=keep_candidates,
        kind_label="interior",
    )

    if back_list:
        _ensure_prompt_images(
            prompts=back_list,
            images_dir=images_dir,
            provider=provider,
            optimizer=optimizer,
            ref_sheets=ref_sheets,
            ref_description_string=ref_description_string,
            overlay_styles=overlay_styles,
            overlays_enabled=overlays_enabled,
            cand_n=cand_n,
            image_provider_mode=image_provider_mode,
            keep_candidates=keep_candidates,
            kind_label="back-matter",
        )

    interior_pdf = output_dir / "book" / "interior.pdf"
    build_interior_pdf(images_dir, page_prompts, pipeline_cfg, interior_pdf)

    cover_pdf = output_dir / "book" / "cover.pdf"
    build_cover_pdf(
        page_prompts=page_prompts,
        pipeline_cfg=pipeline_cfg,
        images_dir=images_dir,
        out_pdf=cover_pdf,
        provider=provider,
        optimizer=optimizer,
        ref_sheets=ref_sheets,
        ref_description_string=ref_description_string,
        overlay_styles=overlay_styles,
        image_provider_mode=image_provider_mode,
        cand_n=cand_n,
        keep_candidates=keep_candidates,
        repo_root=config_dir.parent,
    )

    return {
        "interior_pdf": str(interior_pdf),
        "cover_pdf": str(cover_pdf),
        "images_dir": str(images_dir),
        "refs_used": [p.name for p in ref_sheets],
        "front_matter_count": len(front_list),
        "interior_count": len(interior_list),
        "back_matter_count": len(back_list),
        "overlays_enabled": overlays_enabled,
        "one_sided": bool(pipeline_cfg.get("one_sided", True)),
        "blank_after_interior": bool(pipeline_cfg.get("blank_after_interior", True)),
        "pad_to_multiple_of_4": bool(pipeline_cfg.get("pad_to_multiple_of_4", True)),
        "dpi": int(pipeline_cfg.get("dpi", 300)),
    }
