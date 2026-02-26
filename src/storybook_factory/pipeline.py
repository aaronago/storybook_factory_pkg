# pipeline.py
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from PIL import Image

from .image_provider import ImageProvider, ensure_dir
from .overlay_renderer import TextStyle, apply_overlays
from .prompt_optimizer import PromptOptimizer

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
    """
    Character sheets live at: <output_dir>/images/refs/*.png
    """
    refs_dir = images_dir / "refs"
    if not refs_dir.exists():
        return []
    return sorted([p for p in refs_dir.glob("*.png") if p.is_file()])


# -----------------------------
# Overlay helpers
# -----------------------------


def _build_overlay_styles(page_prompts: dict[str, Any]) -> dict[str, TextStyle]:
    """
    page_prompts.json contains:
      overlay_styles: { "title": {...}, "body": {...} }
    """
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
    }

    for k, v in raw.items():
        if not isinstance(v, dict):
            raise ValueError(f"overlay_styles['{k}'] must be a dict")
        # Filter to only valid TextStyle parameters
        filtered = {key: val for key, val in v.items() if key in valid_keys}
        styles[k] = TextStyle(**filtered)
    return styles


def _apply_item_overlays_if_any(
    *,
    images_dir: Path,
    item: dict[str, Any],
    overlay_styles: dict[str, TextStyle],
) -> None:
    """
    If item has overlays: [...] apply them onto the final image file in-place.
    """
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

    # For coloring book interiors, keep grayscale
    out.convert("L").save(img_path)


# -----------------------------
# PDF assembly (pure)
# -----------------------------


def build_interior_pdf(
    images_dir: Path,
    page_prompts: dict[str, Any],
    pipeline_cfg: dict[str, Any],
    out_pdf: Path,
) -> Path:
    """
    Assemble a grayscale interior PDF from already-generated images.

    Ordering:
      1) front_matter_prompts (as listed)
      2) interior_prompts sorted by page
      3) back_matter_prompts (as listed)
    """
    images: list[Image.Image] = []

    def _load_grayscale(path: Path) -> Image.Image:
        return Image.open(path).convert("L")

    front = _as_list_of_dicts(
        page_prompts.get("front_matter_prompts"), "front_matter_prompts"
    )
    interior = _sorted_interior_prompts(page_prompts)
    back = _as_list_of_dicts(
        page_prompts.get("back_matter_prompts"), "back_matter_prompts"
    )

    # Keep front/back order stable. If you provide "page" fields, we sort by them.
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

    for fm in front:
        fname = fm.get("file")
        if not fname:
            raise ValueError(f"Front matter item missing 'file': {fm}")
        p = images_dir / fname
        if not p.exists():
            raise FileNotFoundError(f"Missing front-matter image: {p}")
        images.append(_load_grayscale(p))

    for pmt in interior:
        fname = pmt["file"]
        page_num = pmt["page"]
        p = images_dir / fname
        if not p.exists():
            raise FileNotFoundError(f"Missing interior page {page_num} image: {p}")
        images.append(_load_grayscale(p))

    for bm in back:
        fname = bm.get("file")
        if not fname:
            raise ValueError(f"Back matter item missing 'file': {bm}")
        p = images_dir / fname
        if not p.exists():
            raise FileNotFoundError(f"Missing back-matter image: {p}")
        images.append(_load_grayscale(p))

    if not images:
        raise RuntimeError("No images found; cannot build interior PDF.")

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
    overlay_styles: dict[str, TextStyle],
    cand_n: int,
    image_provider_mode: str,
    keep_candidates: bool,
    kind_label: str,
) -> None:
    """
    Ensure images exist for a prompt list (front/interior/back) in new schema.
    Applies overlays in-place after generation/copy/placeholder if overlays exist.
    """

    # Avoid “kids appear on title page” by NOT attaching ref sheets for front/back matter.
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

        # If file exists, still apply overlays (in case overlays were added later)
        if final_path.exists():
            _apply_item_overlays_if_any(
                images_dir=images_dir, item=item, overlay_styles=overlay_styles
            )
            print(f"[pipeline] skipping existing {kind_label} image: {fname}")
            continue

        # Folder mode: copy from assets if possible; otherwise placeholder
        if image_provider_mode == "folder":
            got = provider._copy_from_assets(fname)
            if got is None:
                provider._placeholder(
                    fname, f"[MISSING FILE] {raw_prompt}", cover=False
                )
            _apply_item_overlays_if_any(
                images_dir=images_dir, item=item, overlay_styles=overlay_styles
            )
            continue

        # Non-gpt modes: placeholder
        if image_provider_mode != "gpt-image":
            provider._placeholder(fname, raw_prompt, cover=False)
            _apply_item_overlays_if_any(
                images_dir=images_dir, item=item, overlay_styles=overlay_styles
            )
            continue

        # gpt-image mode
        assert optimizer is not None

        page_title = title
        if isinstance(page_num, int):
            page_title = f"{title} (page {page_num})"

        optimized = optimizer.optimize(raw_prompt, page_title=page_title)

        # Only add ref-sheet instruction for interior pages
        if use_refs and ref_sheets:
            optimized += "\n\nUse the attached character sheet reference images as the canonical identity and style."

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
        provider.finalize_candidate(candidate_path=best_candidate, final_filename=fname)

        # Apply overlays AFTER the final file is in place
        _apply_item_overlays_if_any(
            images_dir=images_dir, item=item, overlay_styles=overlay_styles
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
    openai_interior_model: str | None = None,
    dry_run: bool = False,
    image_quality: str = "standard",
) -> dict[str, Any]:
    """
    Pipeline expects config_dir contains:
      - page_prompts.json
      - pipeline_config.json
    """
    interior_model = openai_interior_model or openai_model

    config_dir = config_dir.resolve()
    output_dir = output_dir.resolve()
    assets_dir = assets_dir.resolve()
    ensure_dir(output_dir)

    images_dir = output_dir / "images"
    ensure_dir(images_dir)

    page_prompts = load_json(config_dir / "page_prompts.json")
    pipeline_cfg = load_json(config_dir / "pipeline_config.json")

    # Image sizes
    ip = pipeline_cfg.get("interior_pixels", {"w": 1024, "h": 1536})
    interior_px = (int(ip["w"]), int(ip["h"]))

    provider = ImageProvider(
        out_dir=images_dir,
        interior_px=interior_px,
        cover_px=interior_px,  # unused in this interior builder
        mode=image_provider_mode,
        openai_cover_model=None,
        openai_interior_model=interior_model,
        assets_dir=assets_dir,
        dry_run=dry_run,
        image_quality=image_quality,
    )

    optimizer: PromptOptimizer | None = None
    if image_provider_mode == "gpt-image":
        optimizer = PromptOptimizer.from_env()

    # HARD CAP candidates to <= 2
    cand_n = min(int(os.getenv("STORYBOOK_CANDIDATES_N", "2")), 2)

    keep_candidates = os.getenv("STORYBOOK_KEEP_CANDIDATES", "").lower() in {
        "1",
        "true",
        "yes",
    }

    # Load character sheets once
    ref_sheets = _load_reference_sheets(images_dir)
    if ref_sheets:
        print(
            f"[pipeline] loaded {len(ref_sheets)} character sheets from {images_dir / 'refs'}"
        )
    else:
        print(
            "[pipeline] WARNING: no character sheets found in images/refs — pages may drift"
        )

    # Build overlay styles from JSON (compiled from YAML)
    overlay_styles = _build_overlay_styles(page_prompts)

    # Prompt groups (NEW schema)
    front_list = _as_list_of_dicts(
        page_prompts.get("front_matter_prompts"), "front_matter_prompts"
    )
    interior_list = _sorted_interior_prompts(page_prompts)
    back_list = _as_list_of_dicts(
        page_prompts.get("back_matter_prompts"), "back_matter_prompts"
    )

    # Generate images in order:
    if front_list:
        _ensure_prompt_images(
            prompts=front_list,
            images_dir=images_dir,
            provider=provider,
            optimizer=optimizer,
            ref_sheets=ref_sheets,
            overlay_styles=overlay_styles,
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
        overlay_styles=overlay_styles,
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
            overlay_styles=overlay_styles,
            cand_n=cand_n,
            image_provider_mode=image_provider_mode,
            keep_candidates=keep_candidates,
            kind_label="back-matter",
        )

    # Assemble final interior PDF
    interior_pdf = output_dir / "book" / "interior.pdf"
    build_interior_pdf(images_dir, page_prompts, pipeline_cfg, interior_pdf)

    return {
        "interior_pdf": str(interior_pdf),
        "images_dir": str(images_dir),
        "refs_used": [p.name for p in ref_sheets],
        "front_matter_count": len(front_list),
        "interior_count": len(interior_list),
        "back_matter_count": len(back_list),
        "overlay_style_keys": sorted(list(overlay_styles.keys())),
    }
