# pipeline.py
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from PIL import Image

from .image_provider import ImageProvider, ensure_dir
from .prompt_optimizer import PromptOptimizer

DEFAULT_RUBRIC = {
    "must_have": [
        "Black-and-white line art only (no color).",
        "No shading, no gray tones, no gradients, no shadows.",
        "No large solid filled black areas.",
        "Characters must match the provided character bible (no generic substitutions).",
        "No extra animals or extra people not in the cast.",
        "Composition is clear and readable for a child's coloring page.",
    ],
    "should_have": [
        "Line hierarchy: thicker outer contours for main subjects, thinner for background.",
        "Off-center or asymmetrical composition (avoid perfectly centered, stock layouts).",
        "One clear focal point and generous white space for coloring.",
        "Kid-friendly shapes with uncluttered background.",
    ],
    "penalize": [
        "Cross-hatching or heavy texture that looks like shading.",
        "Busy backgrounds that reduce colorable white space.",
        "Random extra props that distract from the scene.",
    ],
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _load_reference_sheets(images_dir: Path) -> list[Path]:
    """
    Character sheets live at: <output_dir>/images/refs/*.png

    We attach ALL of them to every page generation. That’s the simplest “always consistent”
    approach and matches your plan of sending them with every page prompt.
    """
    refs_dir = images_dir / "refs"
    if not refs_dir.exists():
        return []
    return sorted([p for p in refs_dir.glob("*.png") if p.is_file()])


def build_interior_pdf(
    images_dir: Path,
    page_prompts: dict[str, Any],
    pipeline_cfg: dict[str, Any],
    out_pdf: Path,
) -> Path:
    images: list[Image.Image] = []

    front_matter = page_prompts.get("front_matter", {})
    for key in ["title_page", "dedication_page"]:
        fm = front_matter.get(key)
        if not fm:
            continue
        img_path = images_dir / fm["file"]
        if not img_path.exists():
            raise FileNotFoundError(f"Missing front-matter image: {img_path}")
        images.append(Image.open(img_path).convert("L"))

    for p in sorted(page_prompts["interior_prompts"], key=lambda x: x["page"]):
        img_path = images_dir / p["file"]
        if not img_path.exists():
            raise FileNotFoundError(f"Missing interior page image: {img_path}")
        images.append(Image.open(img_path).convert("L"))

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


def _regen_tweak_prompt(prompt: str, attempt: int) -> str:
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
    openai_interior_model: str | None = None,
    dry_run: bool = False,
    image_quality: str = "standard",
) -> dict[str, Any]:
    interior_model = openai_interior_model or openai_model

    config_dir = config_dir.resolve()
    output_dir = output_dir.resolve()
    assets_dir = assets_dir.resolve()
    ensure_dir(output_dir)

    images_dir = output_dir / "images"
    ensure_dir(images_dir)

    page_prompts = load_json(config_dir / "page_prompts.json")
    pipeline_cfg = load_json(config_dir / "pipeline_config.json")

    ip = pipeline_cfg.get("interior_pixels", {"w": 1024, "h": 1536})

    provider = ImageProvider(
        out_dir=images_dir,
        interior_px=(ip["w"], ip["h"]),
        cover_px=(ip["w"], ip["h"]),  # unused
        mode=image_provider_mode,
        openai_cover_model=None,
        openai_interior_model=interior_model,
        assets_dir=assets_dir,
        dry_run=dry_run,
        image_quality=image_quality,
    )

    optimizer: PromptOptimizer | None = None
    # reviewer: ImageReviewer | None = None  # ImageReviewer not currently available
    if image_provider_mode == "gpt-image":
        optimizer = PromptOptimizer.from_env()
        # reviewer = ImageReviewer.from_env()

    # HARD CAP candidates to <= 2 (your requirement)
    cand_n = min(int(os.getenv("STORYBOOK_CANDIDATES_N", "2")), 2)
    # max_regen = int(os.getenv("STORYBOOK_MAX_REGEN", "2"))  # not used without reviewer
    keep_candidates = os.getenv("STORYBOOK_KEEP_CANDIDATES", "").lower() in {
        "1",
        "true",
        "yes",
    }
    # rubric = DEFAULT_RUBRIC  # not used without reviewer

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

    # 1) Front matter (optional)
    front_matter = page_prompts.get("front_matter", {})
    if front_matter:
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

            # Small nudge: tell the model what the refs are for
            optimized += "\n\nUse the attached character sheet reference images as the canonical identity and style."

            print(f"[pipeline] generating front-matter page: {key}")
            cands = provider.generate_candidates(
                base_filename=fname,
                prompt=optimized,
                cover=False,
                n=cand_n,
                reference_images=ref_sheets,
            )

            provider.finalize_candidate(candidate_path=cands[0], final_filename=fname)
            provider.cleanup_candidates(base_filename=fname)

    # 2) Interior pages
    for p in sorted(page_prompts["interior_prompts"], key=lambda x: x["page"]):
        fname = p["file"]
        title = p.get("title")
        raw_prompt = p["prompt"]

        final_path = images_dir / fname
        if final_path.exists():
            print(f"[pipeline] skipping existing interior image: {fname}")
            continue

        if image_provider_mode == "folder":
            got = provider._copy_from_assets(fname)
            if got is not None:
                continue
            provider._placeholder(fname, f"[MISSING FILE] {raw_prompt}", cover=False)
            continue

        if image_provider_mode != "gpt-image":
            provider._placeholder(fname, raw_prompt, cover=False)
            continue

        assert optimizer is not None

        optimized = optimizer.optimize(raw_prompt, page_title=title)
        optimized += "\n\nUse the attached character sheet reference images as the canonical identity and style."

        print(f"[pipeline] generating candidates page {p['page']:02d} ({title})")
        cands = provider.generate_candidates(
            base_filename=fname,
            prompt=optimized,
            cover=False,
            n=cand_n,
            reference_images=ref_sheets,
        )

        # Just pick the first candidate (no review available)
        best_candidate = cands[0]
        best_reason = "(no review available)"

        print(f"[pipeline] selected candidate 0 of {len(cands)} reason={best_reason}")

        provider.finalize_candidate(candidate_path=best_candidate, final_filename=fname)

        if keep_candidates:
            # Keep candidate records for reference
            report = {
                "page": p["page"],
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

    interior_pdf = output_dir / "book" / "interior.pdf"
    build_interior_pdf(images_dir, page_prompts, pipeline_cfg, interior_pdf)

    return {
        "interior_pdf": str(interior_pdf),
        "images_dir": str(images_dir),
        "refs_used": [p.name for p in ref_sheets],
    }
