# cli.py
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv

from .character_sheets import (
    discover_character_refs_from_assets,
    make_reference_sheets,
    parse_character_ids,
)
from .generator import generate_from_brief
from .image_provider import ImageProvider
from .pipeline import run_pipeline
from .settings import settings

load_dotenv()

print(settings.summary())


def _copy_refs(refs_source: Path, dest_refs_dir: Path) -> int:
    """
    Copy all .png refs from:
      <refs_source>/images/refs/*.png
    into:
      <dest_output>/images/refs/
    Returns count copied.
    """
    src_refs_dir = refs_source / "images" / "refs"
    if not src_refs_dir.exists():
        print(f"[build] refs-source has no refs dir: {src_refs_dir}")
        return 0

    dest_refs_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for p in sorted(src_refs_dir.glob("*.png")):
        if p.is_file():
            shutil.copy2(p, dest_refs_dir / p.name)
            count += 1
    return count


def _ensure_refs_for_build(
    *,
    output_dir: Path,
    assets_dir: Path,
    image_provider: str,
    openai_interior_model: str,
    image_quality: str,
    config_dir: Path | None,
    dry_run: bool,
    refs_source: Path | None,
) -> None:
    """
    Ensure <output_dir>/images/refs contains the character sheets that the pipeline will use.
    Priority:
      1) If refs_source provided -> copy them in.
      2) If refs already exist in output -> keep them.
      3) Else -> generate refs from assets/characters/* into output.
    """
    images_dir = output_dir / "images"
    refs_dir = images_dir / "refs"
    images_dir.mkdir(parents=True, exist_ok=True)
    refs_dir.mkdir(parents=True, exist_ok=True)

    # 1) Copy from previous run if requested
    if refs_source is not None:
        copied = _copy_refs(refs_source.resolve(), refs_dir)
        if copied:
            print(f"[build] copied {copied} reference sheet(s) from {refs_source}")
            return
        print("[build] refs-source provided but nothing copied; falling back...")

    # 2) If refs exist already, do nothing
    existing = sorted(refs_dir.glob("*.png"))
    if existing:
        print(
            f"[build] found {len(existing)} existing reference sheet(s) in {refs_dir}"
        )
        return

    # 3) Generate refs into this build output
    provider = ImageProvider(
        out_dir=images_dir,
        interior_px=(1024, 1536),
        cover_px=(1024, 1536),  # unused
        mode=image_provider,
        openai_interior_model=openai_interior_model,
        openai_cover_model=None,
        assets_dir=assets_dir,
        dry_run=dry_run,
        image_quality=image_quality,
    )

    subjects = discover_character_refs_from_assets(assets_dir)

    # Optional: filter by IDs listed in page_prompts.json (NO PATHS)
    if config_dir is not None:
        try:
            pp = config_dir / "page_prompts.json"
            if pp.exists():
                page_prompts = json.loads(pp.read_text())
                ids = parse_character_ids(page_prompts)
                if ids:
                    allow = {s.lower() for s in ids}
                    subjects = [s for s in subjects if s.id.lower() in allow]
        except Exception:
            pass

    if not subjects:
        print("[build] WARNING: no subjects discovered; continuing without refs")
        return

    out_map = make_reference_sheets(
        provider=provider,
        subjects=subjects,
        refs_dir=refs_dir,
        force=False,
    )

    print("[build] generated character sheets:")
    for k, v in out_map.items():
        print(f"  {k}: {v}")


def main():
    parser = argparse.ArgumentParser(prog="storybook-factory")
    subparsers = parser.add_subparsers(dest="command")

    # -----------------------
    # brief2json command
    # -----------------------
    p1 = subparsers.add_parser(
        "brief2json",
        help=(
            "Convert a human YAML story brief + theme pack into "
            "page_prompts.json and visual_bible.json"
        ),
    )
    p1.add_argument("--brief", required=True, help="Path to brief YAML")
    p1.add_argument(
        "--theme",
        required=True,
        help="Name of theme/scene-pack YAML (no extension)",
    )
    p1.add_argument(
        "--out",
        required=True,
        help="Output directory for generated JSON",
    )

    # -----------------------
    # build command (NO COVERS)
    # -----------------------
    p2 = subparsers.add_parser(
        "build",
        help="Generate character sheets + interior images + interior PDF + ZIP (no covers)",
    )
    p2.add_argument("--config-dir", required=True, help="Config directory (JSON files)")
    p2.add_argument(
        "--output-dir", required=True, help="Output directory for artifacts"
    )
    p2.add_argument(
        "--assets-dir",
        default="assets",
        help="Folder with static assets (expects assets/characters/<id>/... for refs)",
    )
    p2.add_argument(
        "--image-provider",
        choices=["mock", "folder", "gpt-image"],
        default="mock",
        help="Image generation backend",
    )
    p2.add_argument(
        "--openai-interior-model",
        default="dall-e-2",
        help="OpenAI model for interior pages + character sheets (default: dall-e-2)",
    )
    p2.add_argument(
        "--image-quality",
        choices=["low", "medium", "high", "auto"],
        default="low",
        help="Image quality for OpenAI generation (default: low)",
    )
    p2.add_argument(
        "--dry-run",
        action="store_true",
        help="Print prompts without generating images",
    )
    p2.add_argument(
        "--refs-source",
        required=False,
        help=(
            "Optional: path to a previous output dir that already has images/refs/*.png. "
            "Those refs will be copied into this build output before generating pages."
        ),
    )

    # -----------------------
    # frontmatter command (NO COVERS)
    # -----------------------
    p3 = subparsers.add_parser(
        "frontmatter",
        help="Generate ONLY front-matter pages (skip interiors; no covers exist)",
    )
    p3.add_argument("--config-dir", required=True, help="Config directory (JSON files)")
    p3.add_argument(
        "--output-dir", required=True, help="Output directory for artifacts"
    )
    p3.add_argument(
        "--assets-dir",
        default="assets",
        help="Folder with any static assets (if used by pipeline)",
    )
    p3.add_argument(
        "--image-provider",
        choices=["mock", "folder", "gpt-image"],
        default="mock",
        help="Image generation backend",
    )
    p3.add_argument(
        "--openai-interior-model",
        default="dall-e-2",
        help="OpenAI model for front-matter pages (default: dall-e-2)",
    )

    # -----------------------
    # refs command
    # -----------------------
    p0 = subparsers.add_parser(
        "refs",
        help="Generate ONLY character reference sheets (no interiors, no PDF)",
    )
    p0.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for reference sheets",
    )
    p0.add_argument(
        "--assets-dir",
        default="assets",
        help="Assets folder (expects assets/characters/<id>/...)",
    )
    p0.add_argument(
        "--image-provider",
        choices=["mock", "folder", "gpt-image"],
        default="gpt-image",
        help="Image generation backend",
    )
    p0.add_argument(
        "--openai-interior-model",
        default="gpt-image-1",
        help="OpenAI model for character sheets",
    )
    p0.add_argument(
        "--config-dir",
        required=False,
        help="Optional config dir containing page_prompts.json (for character ID filtering only)",
    )

    args = parser.parse_args()

    # -----------------------
    # Handle commands
    # -----------------------
    if args.command == "brief2json":
        brief = Path(args.brief)
        theme_name = args.theme
        theme_path = (
            Path(__file__).resolve().parent.parent
            / "scene_packs"
            / f"{theme_name}.yaml"
        )

        if not theme_path.exists():
            print(f"Scene pack not found: {theme_path}")
            sys.exit(1)

        out_dir = Path(args.out)
        generate_from_brief(brief, theme_path, out_dir)
        print(f"Generated JSON config in: {out_dir}")
        return

    if args.command == "refs":
        output_dir = Path(args.output_dir).resolve()
        assets_dir = Path(args.assets_dir).resolve()

        images_dir = output_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        provider = ImageProvider(
            out_dir=images_dir,
            interior_px=(1024, 1536),
            cover_px=(1024, 1536),  # unused
            mode=args.image_provider,
            openai_interior_model=args.openai_interior_model,
            openai_cover_model=None,
            assets_dir=assets_dir,
            dry_run=False,
        )

        # 1) Discover all subjects from assets_dir/characters/*
        subjects = discover_character_refs_from_assets(assets_dir)

        # 2) Optional: filter by IDs listed in page_prompts.json (NO PATHS)
        if getattr(args, "config_dir", None):
            try:
                cfg_dir = Path(args.config_dir).resolve()
                pp = cfg_dir / "page_prompts.json"
                if pp.exists():
                    page_prompts = json.loads(pp.read_text())
                    ids = parse_character_ids(page_prompts)
                    if ids:
                        allow = set(ids)
                        subjects = [s for s in subjects if s.id.lower() in allow]
            except Exception:
                pass

        refs_dir = images_dir / "refs"
        out_map = make_reference_sheets(
            provider=provider,
            subjects=subjects,
            refs_dir=refs_dir,
            force=False,
        )

        print("Generated character sheets:")
        for k, v in out_map.items():
            print(f"  {k}: {v}")

        return

    elif args.command == "build":
        config_dir = Path(args.config_dir).resolve()
        output_dir = Path(args.output_dir).resolve()
        assets_dir = Path(args.assets_dir).resolve()

        refs_source = Path(args.refs_source).resolve() if args.refs_source else None

        # Ensure refs exist IN THIS output folder before generating pages
        _ensure_refs_for_build(
            output_dir=output_dir,
            assets_dir=assets_dir,
            image_provider=args.image_provider,
            openai_interior_model=args.openai_interior_model,
            image_quality=args.image_quality,
            config_dir=config_dir,
            dry_run=args.dry_run,
            refs_source=refs_source,
        )

        result = run_pipeline(
            config_dir=config_dir,
            output_dir=output_dir,
            assets_dir=assets_dir,
            image_provider_mode=args.image_provider,
            openai_interior_model=args.openai_interior_model,
            dry_run=args.dry_run,
            image_quality=args.image_quality,
        )
        print(result)
        return

    elif args.command == "frontmatter":
        result = run_pipeline(
            config_dir=Path(args.config_dir),
            output_dir=Path(args.output_dir),
            assets_dir=Path(args.assets_dir),
            image_provider_mode=args.image_provider,
            openai_interior_model=args.openai_interior_model,
            dry_run=False,
        )
        print(result)
        return

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
