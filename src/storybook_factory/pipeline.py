# pipeline.py
from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

from PIL import Image

from .image_provider import ImageProvider, ensure_dir
from .image_reviewer import DEFAULT_RUBRIC, ImageReviewer
from .prompt_optimizer import PromptOptimizer

print("STORYBOOK_KEEP_CANDIDATES =", os.getenv("STORYBOOK_KEEP_CANDIDATES"))


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def make_contact_sheet(candidates, out_path):
    """
    Creates a horizontal contact sheet of candidate images for visual inspection.
    Used only in debug / tuning mode.
    """
    imgs = [Image.open(p).convert("RGB") for p in candidates]

    if not imgs:
        return

    w, h = imgs[0].size
    sheet = Image.new("RGB", (w * len(imgs), h), "white")

    for i, img in enumerate(imgs):
        sheet.paste(img, (i * w, 0))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def build_interior_pdf(
    images_dir: Path,
    page_prompts: dict[str, Any],
    pipeline_cfg: dict[str, Any],
    out_pdf: Path,
) -> Path:
    images: list[Image.Image] = []
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
    pipeline_cfg: dict[str, Any],
    covers: dict[str, Any],
    out_pdf: Path,
    interior_page_count: int,
) -> Path:
    dpi = pipeline_cfg["dpi"]
    trim_w = pipeline_cfg["trim_in"]["w"]
    trim_h = pipeline_cfg["trim_in"]["h"]
    bleed = 0.125

    side_w_in = trim_w + 2 * bleed
    side_h_in = trim_h + 2 * bleed
    side_w_px = int(round(side_w_in * dpi))
    side_h_px = int(round(side_h_in * dpi))

    total_w_px = side_w_px * 2
    total_h_px = side_h_px

    canvas = Image.new("RGB", (total_w_px, total_h_px), "white")

    front_key = None
    back_key = None
    for k in covers.keys():
        lk = k.lower()
        if "front" in lk:
            front_key = k
        elif "back" in lk:
            back_key = k

    if front_key is None or back_key is None:
        keys = list(covers.keys())
        if len(keys) < 2:
            raise ValueError("Need at least two cover entries to build wrap cover.")
        back_key, front_key = keys[0], keys[1]

    front_f = covers[front_key].get("file")
    back_f = covers[back_key].get("file")
    if not front_f or not back_f:
        raise ValueError(
            "Both front and back cover entries must provide a 'file' field."
        )

    front_img_path = images_dir / front_f
    back_img_path = images_dir / back_f
    if not front_img_path.exists():
        raise FileNotFoundError(f"Front cover image not found: {front_img_path}")
    if not back_img_path.exists():
        raise FileNotFoundError(f"Back cover image not found: {back_img_path}")

    front_img = Image.open(front_img_path).convert("RGB").resize((side_w_px, side_h_px))
    back_img = Image.open(back_img_path).convert("RGB").resize((side_w_px, side_h_px))

    canvas.paste(back_img, (0, 0))
    canvas.paste(front_img, (side_w_px, 0))

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_pdf, "PDF", resolution=dpi)
    return out_pdf


def build_package(
    root: Path,
    interior_pdf: Path | None,
    cover_pdf: Path | None,
    images_dir: Path,
    out_zip: Path,
) -> Path:
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


def _regen_tweak_prompt(prompt: str, attempt: int) -> str:
    """
    Small deterministic tweaks when all candidates fail must-have checks.
    Keep it minimal and consistent.
    """
    tweaks = [
        "Simplify the background further and increase white space.",
        "Remove any shading/cross-hatching; use outlines only.",
        "Use thicker outer contour lines for main subjects; thin lines for background.",
        "Avoid centered symmetry; place the focal element off-center.",
    ]
    extra = tweaks[min(attempt, len(tweaks) - 1)]
    return prompt.strip() + "\n\nIMPORTANT RETRY ADJUSTMENT: " + extra


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

    page_prompts = load_json(config_dir / "page_prompts.json")
    pipeline_cfg = load_json(config_dir / "pipeline_config.json")

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

    # New components (only needed for gpt-image mode)
    optimizer: PromptOptimizer | None = None
    reviewer: ImageReviewer | None = None
    if image_provider_mode == "gpt-image":
        optimizer = PromptOptimizer.from_env()
        reviewer = ImageReviewer.from_env()

    # Settings
    cand_n = int(os.getenv("STORYBOOK_CANDIDATES_N", "4"))
    max_regen = int(os.getenv("STORYBOOK_MAX_REGEN", "2"))
    do_edit_pass = os.getenv("STORYBOOK_ENABLE_EDIT_PASS", "").lower() in {
        "1",
        "true",
        "yes",
    }
    keep_candidates = os.getenv("STORYBOOK_KEEP_CANDIDATES", "").lower() in {
        "1",
        "true",
        "yes",
    }
    rubric = DEFAULT_RUBRIC

    # 1) Generate interior images
    if not covers_only:
        for p in sorted(page_prompts["interior_prompts"], key=lambda x: x["page"]):
            fname = p["file"]
            title = p.get("title")
            raw_prompt = p["prompt"]

            final_path = images_dir / fname
            if final_path.exists():
                print(f"[pipeline] skipping existing interior image: {fname}")
                continue

            # Folder mode fallback
            if image_provider_mode == "folder":
                got = provider._copy_from_assets(fname)  # intentional reuse
                if got is not None:
                    continue
                provider._placeholder(
                    fname, f"[MISSING FILE] {raw_prompt}", cover=False
                )
                continue

            if image_provider_mode != "gpt-image":
                provider._placeholder(fname, raw_prompt, cover=False)
                continue

            assert optimizer is not None and reviewer is not None

            # Optimize prompt (UI-like boost)
            optimized = optimizer.optimize(raw_prompt, page_title=title)

            attempt = 0
            best_candidate: Path | None = None
            best_reason = ""

            while attempt <= max_regen:
                prompt_for_attempt = (
                    optimized
                    if attempt == 0
                    else _regen_tweak_prompt(optimized, attempt)
                )

                print(
                    f"[pipeline] generating candidates for page {p['page']:02d} ({title}) attempt={attempt}"
                )
                cands = provider.generate_candidates(
                    base_filename=fname,
                    prompt=prompt_for_attempt,
                    cover=False,
                    n=cand_n,
                )
                presented = list(cands)
                random.shuffle(presented)

                review = reviewer.review(
                    candidates=presented,
                    page_prompt=prompt_for_attempt,
                    rubric=rubric,
                    page_title=title,
                )

                best_candidate = cands[review.best_index]
                best_reason = review.reasons[review.best_index]

                print(
                    f"[pipeline] review: best_index={review.best_index} "
                    f"score={review.scores[review.best_index]:.1f} "
                    f"needs_regen={review.needs_regen} reason={best_reason}"
                )

                if not review.needs_regen:
                    break

                attempt += 1

            if best_candidate is None:
                # Fallback: should never happen, but keep pipeline resilient.
                cands = provider.generate_candidates(
                    base_filename=fname, prompt=optimized, cover=False, n=1
                )
                best_candidate = cands[0]

            # Optional edit pass
            chosen = best_candidate
            if do_edit_pass and reviewer is not None:
                # If reviewer gave a suggestion, use it. Otherwise skip.
                # (We don't want edits to wander.)
                # Note: last `review` variable is in scope if we reached here through gpt-image mode.
                try:
                    suggestion = review.edit_suggestion  # type: ignore
                except Exception:
                    suggestion = None

                if suggestion:
                    edit_prompt = (
                        "Edit this coloring-book line art to strictly satisfy:\n"
                        "- outlines only, no shading or gray\n"
                        "- remove cross-hatching or shadow texture\n"
                        "- simplify background clutter\n"
                        "- maintain the same characters and scene\n\n"
                        f"Specific fix: {suggestion}"
                    )
                    print(f"[pipeline] applying edit pass for {fname}: {suggestion}")
                    edited_name = Path(fname).stem + "__edited.png"
                    chosen = provider.apply_edit(
                        input_image=chosen,
                        prompt=edit_prompt,
                        out_filename=edited_name,
                        cover=False,
                    )

            # Finalize to expected filename
            provider.finalize_candidate(candidate_path=chosen, final_filename=fname)

            # Save review report if KEEP_CANDIDATES
            if keep_candidates:
                report = {
                    "page": p["page"],
                    "title": title,
                    "prompt": prompt_for_attempt,
                    "best_index": review.best_index,
                    "scores": review.scores,
                    "reasons": review.reasons,
                    "needs_regen": review.needs_regen,
                    "edit_suggestion": review.edit_suggestion,
                    "candidates": [c.name for c in cands],
                }
                report_path = images_dir / f"{Path(fname).stem}__review.json"
                report_path.write_text(json.dumps(report, indent=2))
                if not keep_candidates:
                    provider.cleanup_candidates(base_filename=fname)

            # Cleanup edited intermediate if used
            edited_path = images_dir / (Path(fname).stem + "__edited.png")
            if edited_path.exists():
                try:
                    edited_path.unlink()
                except Exception:
                    pass

    # 1.5) Generate front-matter pages (if defined in page_prompts)
    front_matter = page_prompts.get("front_matter", {})
    if front_matter and not covers_only:
        for key, fm in front_matter.items():
            fname = fm.get("file")
            raw_prompt = fm.get("prompt", "")

            if not fname:
                continue

            final_path = images_dir / fname
            if final_path.exists():
                print(f"[pipeline] skipping existing front-matter page: {fname}")
                continue

            if image_provider_mode != "gpt-image":
                provider._placeholder(fname, raw_prompt, cover=False)
                continue

            assert optimizer is not None

            optimized = optimizer.optimize(
                raw_prompt, page_title=key.replace("_", " ").title()
            )

            print(f"[pipeline] generating front-matter page: {key}")
            cands = provider.generate_candidates(
                base_filename=fname,
                prompt=optimized,
                cover=False,
                n=2,
            )

            # No reviewer needed — pick best visually later if desired
            provider.finalize_candidate(candidate_path=cands[0], final_filename=fname)
            provider.cleanup_candidates(base_filename=fname)

    # 2) Generate cover images (simple version, same pattern but usually fewer needs)
    if not interiors_only:
        for key, cov in page_prompts["covers"].items():
            fname = cov.get("file")
            raw_prompt = cov.get("prompt", "")
            if not fname:
                continue

            final_path = images_dir / fname
            if final_path.exists():
                print(f"[pipeline] skipping existing cover image: {fname}")
                continue

            if image_provider_mode == "folder":
                got = provider._copy_from_assets(fname)
                if got is not None:
                    continue
                provider._placeholder(
                    fname, f"[MISSING COVER {key}] {raw_prompt}", cover=True
                )
                continue

            if image_provider_mode != "gpt-image":
                provider._placeholder(fname, raw_prompt, cover=True)
                continue

            # Covers: optimize prompt too (helps a lot)
            assert optimizer is not None
            cover_prompt = optimizer.optimize(raw_prompt, page_title=f"Cover: {key}")

            print(f"[pipeline] generating cover candidates: {fname}")
            cands = provider.generate_candidates(
                base_filename=fname,
                prompt=cover_prompt,
                cover=True,
                n=max(2, cand_n // 2),
            )

            # For now, pick the first if review disabled; otherwise reuse reviewer
            chosen = cands[0]
            if reviewer is not None:
                review = reviewer.review(
                    candidates=cands,
                    page_prompt=cover_prompt,
                    rubric=rubric,
                    page_title=f"Cover: {key}",
                )
                chosen = cands[review.best_index]
                print(
                    f"[pipeline] cover review: best_index={review.best_index} "
                    f"score={review.scores[review.best_index]:.1f} needs_regen={review.needs_regen}"
                )

                # Save review report if KEEP_CANDIDATES
                if keep_candidates:
                    report = {
                        "cover_key": key,
                        "file": fname,
                        "prompt": cover_prompt,
                        "best_index": review.best_index,
                        "scores": review.scores,
                        "reasons": review.reasons,
                        "needs_regen": review.needs_regen,
                        "edit_suggestion": review.edit_suggestion,
                        "candidates": [c.name for c in cands],
                    }
                    report_path = images_dir / f"{Path(fname).stem}__review.json"
                    report_path.write_text(json.dumps(report, indent=2))

            provider.finalize_candidate(candidate_path=chosen, final_filename=fname)
            provider.cleanup_candidates(base_filename=fname)

    interior_pdf: Path | None = None
    cover_pdf: Path | None = None
    package_zip: Path | None = None

    # 3) Build interior PDF
    if not covers_only:
        interior_pdf = output_dir / "book" / "interior.pdf"
        interior_pdf = build_interior_pdf(
            images_dir, page_prompts, pipeline_cfg, interior_pdf
        )

    # 4) Build wrap cover PDF
    if not interiors_only:
        cover_pdf = output_dir / "cover" / "cover_wrap.pdf"
        cover_pdf = build_wrap_cover_pdf(
            images_dir,
            pipeline_cfg,
            page_prompts["covers"],
            cover_pdf,
            interior_page_count=len(page_prompts["interior_prompts"]),
        )

    # 5) Package ZIP
    if not covers_only and not interiors_only:
        package_zip = output_dir / "book_package.zip"
        package_zip = build_package(
            output_dir, interior_pdf, cover_pdf, images_dir, package_zip
        )

    return {
        "interior_pdf": str(interior_pdf) if interior_pdf else None,
        "cover_pdf": str(cover_pdf) if cover_pdf else None,
        "package_zip": str(package_zip) if package_zip else None,
    }
