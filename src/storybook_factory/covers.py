# covers.py
from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from .image_provider import ImageProvider, ensure_dir
from .overlay_renderer import TextStyle, apply_overlays
from .prompt_optimizer import PromptOptimizer


def _resolve_font_paths(
    styles: dict[str, TextStyle], *, repo_root: Path
) -> dict[str, TextStyle]:
    """
    Make relative font paths work no matter where you run from.
    If a style's font_path is relative, try to resolve it against repo_root.
    """
    out: dict[str, TextStyle] = {}
    for k, st in styles.items():
        fp = st.font_path
        if fp and not os.path.isabs(fp):
            cand = (repo_root / fp).resolve()
            if cand.exists():
                # TextStyle is frozen in overlay_renderer.py, so rebuild via dataclasses.replace
                out[k] = replace(st, font_path=str(cand))
            else:
                out[k] = st
        else:
            out[k] = st
    return out


def _cover_wrap_px(
    *,
    dpi: int,
    back_in: float = 8.625,
    front_in: float = 8.625,
    spine_in: float = 0.132,
    height_in: float = 11.25,
) -> tuple[int, int]:
    """
    Lulu perfect-bound full wrap cover size.

    Total width = back + spine + front
    Height includes bleed.
    """
    width_in = back_in + spine_in + front_in

    w = int(round(width_in * dpi))
    h = int(round(height_in * dpi))
    return w, h


def _apply_cover_overlays_in_place(
    *,
    images_dir: Path,
    item: dict[str, Any],
    overlay_styles: dict[str, TextStyle],
) -> None:
    overlays = item.get("overlays")
    if not overlays:
        return
    if not isinstance(overlays, list):
        raise ValueError(f"cover item overlays must be a list. Got: {type(overlays)}")

    img_path = images_dir / item["file"]
    if not img_path.exists():
        raise FileNotFoundError(
            f"Cannot apply cover overlays; missing image: {img_path}"
        )

    img = Image.open(img_path)
    out = apply_overlays(img, overlays=overlays, styles=overlay_styles)

    # COVER: keep full color (RGB). Do NOT convert to grayscale.
    out.convert("RGB").save(img_path, "PNG")


def _ensure_cover_image(
    *,
    key: str,
    item: dict[str, Any],
    images_dir: Path,
    provider: ImageProvider,
    optimizer: PromptOptimizer | None,
    ref_sheets: list[Path],
    ref_description_string: str = "",
    overlay_styles: dict[str, TextStyle],
    cand_n: int,
    image_provider_mode: str,
    keep_candidates: bool,
) -> Path:
    """
    Ensure cover image exists (front/back). Applies overlays (RGB) after generation.

    Conventions:
      - front cover uses reference sheets (identity/style anchoring)
      - back cover does NOT use refs (keeps it generic)
    """
    fname = item.get("file")
    raw_prompt = item.get("prompt", "")
    title = item.get("title") or f"cover-{key}"

    if not fname:
        raise ValueError(f"covers.{key} missing 'file': {item}")
    if not isinstance(raw_prompt, str) or not raw_prompt.strip():
        raise ValueError(f"covers.{key} missing non-empty 'prompt': {item}")

    final_path = images_dir / fname

    # If exists, still apply overlays (in case overlays were added later)
    if final_path.exists():
        _apply_cover_overlays_in_place(
            images_dir=images_dir, item=item, overlay_styles=overlay_styles
        )
        print(f"[covers] skipping existing cover image: {fname}")
        return final_path

    # Folder mode: copy from assets if possible; otherwise placeholder
    if image_provider_mode == "folder":
        got = provider._copy_from_assets(fname)
        if got is None:
            provider._placeholder(fname, f"[MISSING FILE] {raw_prompt}", cover=True)
        _apply_cover_overlays_in_place(
            images_dir=images_dir, item=item, overlay_styles=overlay_styles
        )
        return final_path

    # Non-gpt modes: placeholder
    if image_provider_mode != "gpt-image":
        provider._placeholder(fname, raw_prompt, cover=True)
        _apply_cover_overlays_in_place(
            images_dir=images_dir, item=item, overlay_styles=overlay_styles
        )
        return final_path

    # gpt-image mode
    assert optimizer is not None

    optimized = optimizer.optimize(
        raw_prompt, page_title=f"{title} ({key})", kind="cover"
    )

    # Use refs only for front cover (keeps back generic)
    use_refs = key.lower() == "front"
    if use_refs and ref_sheets:
        ref_msg = "\n\nUse the attached character sheet reference images as the canonical identity and style."
        if ref_description_string:
            ref_msg += f" {ref_description_string}"
        optimized += ref_msg

    print(f"[covers] generating candidates for cover-{key}: {fname} ({title})")
    cands = provider.generate_candidates(
        base_filename=fname,
        prompt=optimized,
        cover=True,
        n=cand_n,
        reference_images=(ref_sheets if use_refs else []),
    )

    best = cands[0]
    provider.finalize_candidate(
        candidate_path=best,
        final_filename=fname,
        cover=True,
    )

    # Apply overlays AFTER final file is in place
    _apply_cover_overlays_in_place(
        images_dir=images_dir, item=item, overlay_styles=overlay_styles
    )

    if keep_candidates:
        report = {
            "kind": "cover",
            "key": key,
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

    return final_path


def _compose_full_wrap(
    *,
    front_img: Image.Image,
    back_img: Image.Image,
    wrap_size_px: tuple[int, int],
) -> Image.Image:
    """
    Compose Lulu full-wrap cover: back on left half, front on right half.

    IMPORTANT:
    - Uses crop-to-fit (no stretching/squishing).
    - If an input image has the wrong aspect ratio, we preserve proportions
      and crop the excess instead of distorting.
    """
    wrap_w, wrap_h = wrap_size_px

    # split width (odd widths get the extra pixel on the front)
    back_w = wrap_w // 2
    front_w = wrap_w - back_w

    # Fit each half WITHOUT distortion (crop-to-fit)
    back_fitted = ImageOps.fit(
        back_img.convert("RGB"),
        (back_w, wrap_h),
        method=Image.LANCZOS,
        centering=(0.5, 0.5),  # center crop
    )
    front_fitted = ImageOps.fit(
        front_img.convert("RGB"),
        (front_w, wrap_h),
        method=Image.LANCZOS,
        centering=(0.5, 0.5),  # center crop
    )

    canvas = Image.new("RGB", (wrap_w, wrap_h), "white")
    canvas.paste(back_fitted, (0, 0))
    canvas.paste(front_fitted, (back_w, 0))
    return canvas


def build_cover_pdf(
    *,
    page_prompts: dict[str, Any],
    pipeline_cfg: dict[str, Any],
    images_dir: Path,
    out_pdf: Path,
    provider: ImageProvider,
    optimizer: PromptOptimizer | None,
    ref_sheets: list[Path],
    ref_description_string: str = "",
    overlay_styles: dict[str, TextStyle],
    image_provider_mode: str,
    cand_n: int,
    keep_candidates: bool,
    # Optional: if you want to persist the composed wrap PNG for inspection
    write_wrap_png: bool = True,
    repo_root: Path | None = None,
) -> Path:
    """
    Builds a Lulu-ready, single-page, flattened cover PDF (full wrap).

    Expects page_prompts.json to include:
      page_prompts["covers"]["front"] -> {file,prompt,title?,overlays?}
      page_prompts["covers"]["back"]  -> {file,prompt,title?}
      page_prompts["covers"]["back_overlays"] optional:
          {enabled: bool, overlays: [...]} or {enabled: bool, ...}

    Notes:
      - Covers stay RGB (full color).
      - Overlays applied to FRONT art image (and optional back blurb).
      - Final output: one-page PDF at 17.25x11.25 inches @ dpi.
    """
    covers = page_prompts.get("covers") or {}
    if not isinstance(covers, dict):
        raise ValueError("page_prompts['covers'] must be a dict if present")

    front_item = covers.get("front")
    back_item = covers.get("back")
    if not isinstance(front_item, dict) or not isinstance(back_item, dict):
        raise ValueError(
            "page_prompts['covers'] must contain dicts for 'front' and 'back'"
        )

    dpi = int(pipeline_cfg.get("dpi", 300))

    # Resolve font paths for overlays (relative font paths are common in your repo)
    root = repo_root or Path.cwd()
    overlay_styles = _resolve_font_paths(overlay_styles, repo_root=root)

    ensure_dir(images_dir)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    # 1) Ensure front/back images exist (+ front overlays)
    front_path = _ensure_cover_image(
        key="front",
        item=front_item,
        images_dir=images_dir,
        provider=provider,
        optimizer=optimizer,
        ref_sheets=ref_sheets,
        ref_description_string=ref_description_string,
        overlay_styles=overlay_styles,
        cand_n=cand_n,
        image_provider_mode=image_provider_mode,
        keep_candidates=keep_candidates,
    )
    back_path = _ensure_cover_image(
        key="back",
        item=back_item,
        images_dir=images_dir,
        provider=provider,
        optimizer=optimizer,
        ref_sheets=ref_sheets,
        ref_description_string=ref_description_string,
        overlay_styles=overlay_styles,
        cand_n=cand_n,
        image_provider_mode=image_provider_mode,
        keep_candidates=keep_candidates,
    )

    # Optional back overlays (blurb) — applied to the BACK image file
    back_ov = covers.get("back_overlays")
    if isinstance(back_ov, dict) and back_ov.get("enabled"):
        overlays = back_ov.get("overlays") or []
        if overlays:
            temp_item = {"file": back_item["file"], "overlays": overlays}
            _apply_cover_overlays_in_place(
                images_dir=images_dir, item=temp_item, overlay_styles=overlay_styles
            )

    # 2) Compose full wrap canvas
    wrap_px = _cover_wrap_px(dpi=300, spine_in=0.132)

    front_img = Image.open(front_path).convert("RGB")
    back_img = Image.open(back_path).convert("RGB")
    wrap = _compose_full_wrap(
        front_img=front_img, back_img=back_img, wrap_size_px=wrap_px
    )

    # 3) Persist wrap PNG for inspection (optional)
    if write_wrap_png:
        wrap_png = out_pdf.with_suffix(".png")
        wrap.save(wrap_png, "PNG")
        print(f"[covers] wrote composed wrap PNG: {wrap_png}")

    # 4) Export single-page PDF (flattened)
    # PIL embeds the raster image; overlays are already baked in.
    wrap.save(out_pdf, "PDF", resolution=dpi)
    print(f"[covers] wrote cover PDF: {out_pdf}")
    return out_pdf
