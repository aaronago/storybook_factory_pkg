from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Optional
import json

from PIL import Image

from .image_provider import ImageProvider, ensure_dir


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def build_interior_pdf(
    images_dir: Path,
    page_prompts: Dict[str, Any],
    pipeline_cfg: Dict[str, Any],
    out_pdf: Path,
) -> Path:
    """
    Merge all interior page PNGs into a single interior PDF.
    Pages are ordered by the 'page' field in
    page_prompts["interior_prompts"].
    """
    images: List[Image.Image] = []
    for p in sorted(page_prompts["interior_prompts"], key=lambda x: x["page"]):
        img_path = images_dir / p["file"]
        if not img_path.exists():
            raise FileNotFoundError(f"Missing interior page image: {img_path}")
        im = Image.open(img_path).convert("L")  # grayscale for coloring pages
        images.append(im)

    if not images:
        raise RuntimeError("No interior images found; cannot build interior PDF.")

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    first, *rest = images
    first.save(
        out_pdf,
        "PDF",
        resolution=pipeline_cfg["dpi"],
        save_all=True,
        append_images=rest,
    )
    return out_pdf


def build_wrap_cover_pdf(
    images_dir: Path,
    pipeline_cfg: Dict[str, Any],
    covers: Dict[str, Any],
    out_pdf: Path,
    interior_page_count: int,
) -> Path:
    """
    Build a simple wrap-around cover PDF: back (left) + front (right)
    on a single wide canvas.

    This uses:
      - trim size + bleed from pipeline_cfg
      - two cover images from `covers` (e.g., front_cover / back_cover)
    """
    dpi = pipeline_cfg["dpi"]
    trim_w = pipeline_cfg["trim_in"]["w"]
    trim_h = pipeline_cfg["trim_in"]["h"]
    bleed = 0.125  # in

    # each side: trim + bleed all around
    side_w_in = trim_w + 2 * bleed
    side_h_in = trim_h + 2 * bleed
    side_w_px = int(round(side_w_in * dpi))
    side_h_px = int(round(side_h_in * dpi))

    total_w_px = side_w_px * 2
    total_h_px = side_h_px

    canvas = Image.new("RGB", (total_w_px, total_h_px), "white")

    # we don't care about exact keys; just expect two entries
    # but we preserve your naming convention (front_cover / back_cover, etc.)
    # Prefer front on the right half, back on the left.
    # Try some common key names; fallback to first/second.
    front_key = None
    back_key = None
    for k in covers.keys():
        lk = k.lower()
        if "front" in lk:
            front_key = k
        elif "back" in lk:
            back_key = k

    if front_key is None or back_key is None:
        # fallback: just take first as front, second as back
        keys = list(covers.keys())
        if len(keys) < 2:
            raise ValueError("Need at least two cover entries to build wrap cover.")
        back_key, front_key = keys[0], keys[1]

    front_f = covers[front_key].get("file")
    back_f = covers[back_key].get("file")

    if not front_f or not back_f:
        raise ValueError("Both front and back cover entries must provide a 'file' field.")

    front_img_path = images_dir / front_f
    back_img_path = images_dir / back_f
    if not front_img_path.exists():
        raise FileNotFoundError(f"Front cover image not found: {front_img_path}")
    if not back_img_path.exists():
        raise FileNotFoundError(f"Back cover image not found: {back_img_path}")

    front_img = Image.open(front_img_path).convert("RGB").resize(
        (side_w_px, side_h_px)
    )
    back_img = Image.open(back_img_path).convert("RGB").resize(
        (side_w_px, side_h_px)
    )

    # layout: [ back | front ]
    canvas.paste(back_img, (0, 0))
    canvas.paste(front_img, (side_w_px, 0))

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_pdf, "PDF", resolution=dpi)
    return out_pdf


def build_package(
    root: Path,
    interior_pdf: Optional[Path],
    cover_pdf: Optional[Path],
    images_dir: Path,
    out_zip: Path,
) -> Path:
    """
    Zip up the key deliverables: interior PDF, cover PDF, and all page PNGs.

    If interior_pdf or cover_pdf are None, they are simply omitted from the ZIP.
    """
    import zipfile

    out_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
        if interior_pdf is not None and interior_pdf.exists():
            z.write(interior_pdf, interior_pdf.relative_to(root).as_posix())
        if cover_pdf is not None and cover_pdf.exists():
            z.write(cover_pdf, cover_pdf.relative_to(root).as_posix())
        for img in images_dir.glob("*.png"):
            z.write(img, img.relative_to(root).as_posix())
    return out_zip


def run_pipeline(
    config_dir: Path,
    output_dir: Path,
    assets_dir: Path,
    image_provider_mode: str = "mock",
    openai_model: str | None = None,
    openai_cover_model: str | None = None,
    openai_interior_model: str | None = None,
    dry_run: bool = False,
    covers_only: bool = False,
    interiors_only: bool = False,
) -> str:
    """
    Main orchestration for building the storybook package.

    - config_dir: where page_prompts.json and visual_bible.json live
    - output_dir: where generated images / PDFs will be written
    - assets_dir: extra static assets, if any
    - image_provider_mode: 'mock', 'folder', or 'gpt-image'
    - openai_cover_model: model to use for covers (e.g. gpt-image-1)
    - openai_interior_model: model to use for interiors (e.g. dall-e-2)
    - openai_model: legacy single-model arg; used as fallback if the split args are not provided
    """
    # Backwards-compat: if split models not provided, fall back to openai_model
    cover_model = openai_cover_model or openai_model
    interior_model = openai_interior_model or openai_model
    if covers_only and interiors_only:
        raise ValueError("covers_only and interiors_only cannot both be True.")

    config_dir = config_dir.resolve()
    output_dir = output_dir.resolve()
    assets_dir = assets_dir.resolve()
    ensure_dir(output_dir)
    images_dir = output_dir / "images"
    ensure_dir(images_dir)

    visual_bible = load_json(config_dir / "visual_bible.json")
    page_prompts = load_json(config_dir / "page_prompts.json")
    pipeline_cfg = load_json(config_dir / "pipeline_config.json")

    # Dry run: print prompts and do nothing else
    if dry_run:
        print("\n=== DRY RUN: INTERIOR PROMPTS ===\n")
        for p in sorted(page_prompts["interior_prompts"], key=lambda x: x["page"]):
            print(f"Page {p['page']}: {p['title']}")
            print(p["prompt"])
            print("-" * 60)

        print("\n=== DRY RUN: COVER PROMPTS ===\n")
        for key, cov in page_prompts["covers"].items():
            print(f"{key}: {cov.get('file')}")
            print(cov.get("prompt", ""))
            print("-" * 60)

        print("\n[DRY RUN] No images or PDFs generated.\n")
        return {
            "interior_pdf": None,
            "cover_pdf": None,
            "package_zip": None,
        }

    ip = pipeline_cfg.get("interior_pixels", {"w": 1024, "h": 1536})
    cp = pipeline_cfg.get("cover_pixels", {"w": 1536, "h": 1024})

    provider = ImageProvider(
        out_dir=images_dir,
        interior_px=(ip["w"], ip["h"]),
        cover_px=(cp["w"], cp["h"]),
        mode=image_provider_mode,
        openai_cover_model=cover_model,
        openai_interior_model=interior_model,
        assets_dir=assets_dir,
        dry_run=dry_run,
    )

    # 1) Generate images
    if not covers_only:
        provider.render_interior(page_prompts["interior_prompts"])

    if not interiors_only:
        provider.render_covers(page_prompts["covers"])

    interior_pdf: Optional[Path] = None
    cover_pdf: Optional[Path] = None
    package_zip: Optional[Path] = None

    # 2) Build interior PDF (if requested)
    if not covers_only:
        interior_pdf = output_dir / "book" / "interior.pdf"
        interior_pdf = build_interior_pdf(
            images_dir,
            page_prompts,
            pipeline_cfg,
            interior_pdf,
        )

    # 3) Build wrap-around cover PDF (if requested)
    if not interiors_only:
        cover_pdf = output_dir / "cover" / "cover_wrap.pdf"
        cover_pdf = build_wrap_cover_pdf(
            images_dir,
            pipeline_cfg,
            page_prompts["covers"],
            cover_pdf,
            interior_page_count=len(page_prompts["interior_prompts"]),
        )

    # 4) Build final package ZIP only when we have both interior + cover
    if not covers_only and not interiors_only:
        package_zip = output_dir / "book_package.zip"
        package_zip = build_package(
            output_dir,
            interior_pdf,
            cover_pdf,
            images_dir,
            package_zip,
        )

    return {
        "interior_pdf": str(interior_pdf) if interior_pdf else None,
        "cover_pdf": str(cover_pdf) if cover_pdf else None,
        "package_zip": str(package_zip) if package_zip else None,
    }
