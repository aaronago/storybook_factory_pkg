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
    interior_model: str,
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

    # 3) Generate refs into this build output (REMOVED)
    # The user requested to no longer generate character sheets as a pipeline step.
    # We will only use them if they are already in the output directory or copied via --refs-source.
    print("[build] skipping character sheet generation; continuing...")
    return


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
    p1.add_argument(
        "--assets-dir",
        default="assets",
        help="Folder with static assets (expects assets/characters/<name>/reference/... for refs)",
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
        "--interior-model",
        default="dall-e-2",
        help="Model for interior pages + character sheets (default: dall-e-2)",
    )
    p2.add_argument(
        "--cover-model",
        default=None,
        help="Model for cover pages (default: same as --interior-model)",
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
    p2.add_argument(
        "--output-format",
        choices=["print", "download"],
        default="download",
        help=(
            "Output format profile. "
            "'print' → 300 dpi, one-sided with blank verso pages, padded to multiple of 4 (press-ready). "
            "'download' → 150 dpi, no blank pages, no padding (smaller file for screens/e-readers). "
            "(default: download)"
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
        "--interior-model",
        default="dall-e-2",
        help="Model for front-matter pages (default: dall-e-2)",
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
        "--interior-model",
        default="gpt-image-1",
        help="Model for character sheets",
    )
    p0.add_argument(
        "--config-dir",
        required=False,
        help="Optional config dir containing page_prompts.json (for character ID filtering only)",
    )

    # -----------------------
    # sort-refs command
    # -----------------------
    p_sort = subparsers.add_parser(
        "sort-refs",
        help="Use Gemini to sort reference images into character buckets",
    )
    p_sort.add_argument(
        "--source-dir",
        required=True,
        help="Directory containing the images to sort (e.g. downloads/)",
    )
    p_sort.add_argument(
        "--character",
        action="append",
        required=True,
        help="One or more character names to look for (e.g. --character Alice --character Bob)",
    )
    p_sort.add_argument(
        "--assets-dir",
        default="assets",
        help="Base assets directory (default: assets/)",
    )

    args = parser.parse_args()

    # -----------------------
    # Handle commands
    # -----------------------
    if args.command == "sort-refs":
        import shutil

        from .image_sorter import GeminiImageSorter

        sorter = GeminiImageSorter()
        source_path = Path(args.source_dir)

        # Get all images from source dir
        image_extensions = (".png", ".jpg", ".jpeg", ".webp")
        image_paths = sorted(
            [p for p in source_path.iterdir() if p.suffix.lower() in image_extensions]
        )

        if not image_paths:
            print(f"No images found in {source_path}")
            return

        # print(
        #     f"Sorting {len(image_paths)} images for characters: {', '.join(args.character)}..."
        # )
        results = sorter.sort_user_uploads(image_paths, args.character)

        # Process results and move files
        for character_name, indices in results.items():
            if not indices:
                continue

            # Ensure lower-case bucket for folder name
            bucket_name = character_name.lower().strip()
            # Reference folder: assets/characters/<id>/reference/
            target_dir = (
                Path(args.assets_dir) / "characters" / bucket_name / "reference"
            )
            target_dir.mkdir(parents=True, exist_ok=True)

            for idx in indices:
                if 0 <= idx < len(image_paths):
                    src = image_paths[idx]
                    dest = target_dir / src.name
                    print(f"Moving {src.name} -> {dest}")
                    shutil.move(str(src), str(dest))
        return

    if args.command == "brief2json":
        brief_path = Path(args.brief)
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
        # Use assets_dir from args if available, otherwise default to "assets"
        assets_dir = Path(getattr(args, "assets_dir", "assets")).resolve()

        # Build ref description string & discover reference paths by sorting the assets/characters folder
        ref_description_string = ""

        # We need to discover names from the brief to tell the sorter what to look for
        import yaml

        with open(brief_path) as f:
            brief_data = yaml.safe_load(f)

        char_names = []
        for c in brief_data.get("children", []) or []:
            char_names.append(c.get("name"))
        for p in brief_data.get("pets", []) or []:
            char_names.append(p.get("name"))

        # Also include subfolder names as hints just in case
        chars_root = assets_dir / "characters"
        if chars_root.exists():
            for d in chars_root.iterdir():
                if d.is_dir() and d.name.capitalize() not in char_names:
                    char_names.append(d.name.capitalize())

        # Collect all images from assets/characters/ (non-recursive to avoid processed ones, or recursive)
        # The user wants them from assets/characters primarily.
        image_extensions = (".png", ".jpg", ".jpeg", ".webp")
        all_refs = []
        if chars_root.exists():
            # Get all images in characters/ top level or subfolders if not sorted yet
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
                # print(
                #     f"Sorting {len(all_refs)} reference images for characters: {', '.join(char_names)}..."
                # )
                ref_description_string, sorted_paths = sorter.get_reference_mapping(
                    all_refs, char_names, style_ref_dir=assets_dir / "style_reference"
                )
                if sorted_paths:
                    # print("\n--- Sorted Reference Paths Mapping ---")
                    for i, p in enumerate(sorted_paths):
                        print(f"[{i+1}] {p}")
                    # Note: sorted_paths isn't stored in page_prompts.json but the indices match it.
                    print("---------------------------------------\n")
            except Exception as e:
                print(
                    f"Warning: Gemini sorter failed: {e}. Falling back to folder-based naming."
                )
                # Fallback logic if sorter fails...
                ref_description_string = ""

        generate_from_brief(
            brief_path,
            theme_path,
            out_dir,
            ref_description_string=ref_description_string,
        )
        # print(f"Generated JSON config in: {out_dir}")
        # if ref_description_string:
        #     print(f"Baked references: {ref_description_string}")
        return

    if args.command == "refs":
        output_dir = Path(args.output_dir).resolve()
        assets_dir = Path(args.assets_dir).resolve()

        images_dir = output_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        provider = ImageProvider(
            out_dir=images_dir,
            interior_px=(1024, 1536),
            cover_px=(
                2588,
                3375,
            ),  # unused since we're only generating refs, but set to actual cover size just in case
            mode=args.image_provider,
            interior_model=args.interior_model,
            cover_model=None,
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
            interior_model=args.interior_model,
            image_quality=args.image_quality,
            config_dir=config_dir,
            dry_run=args.dry_run,
            refs_source=refs_source,
        )

        # Build ref description string from the final refs directory
        images_dir = output_dir / "images"
        refs_dir = images_dir / "refs"
        ref_description_string = ""
        if refs_dir.exists():
            from .pipeline import _load_reference_sheets

            ref_sheets = _load_reference_sheets(images_dir)
            if ref_sheets:
                ref_descriptions = []
                for i, ref_path in enumerate(ref_sheets):
                    char_name = None
                    parts = ref_path.parts
                    if "characters" in parts:
                        idx_char = parts.index("characters")
                        if idx_char + 1 < len(parts):
                            char_name = parts[idx_char + 1].capitalize()

                    if char_name:
                        ref_descriptions.append(f"Reference [{i+1}] is {char_name}.")
                    else:
                        ref_descriptions.append(
                            f"Reference [{i+1}] is a style/context reference."
                        )
                ref_description_string = " ".join(ref_descriptions)

        # Inject the ref_description_string into the brief before generation
        # We need to load/save the page_prompts.json with this new info
        # But run_pipeline handles generation if needed.
        # Actually, let's pass it to run_pipeline.

        result = run_pipeline(
            config_dir=config_dir,
            output_dir=output_dir,
            assets_dir=assets_dir,
            image_provider_mode=args.image_provider,
            interior_model=args.interior_model,
            cover_model=args.cover_model,
            dry_run=args.dry_run,
            image_quality=args.image_quality,
            ref_description_string=ref_description_string,
            output_format=args.output_format,
        )
        print(result)
        return

    elif args.command == "frontmatter":
        result = run_pipeline(
            config_dir=Path(args.config_dir),
            output_dir=Path(args.output_dir),
            assets_dir=Path(args.assets_dir),
            image_provider_mode=args.image_provider,
            interior_model=args.interior_model,
            dry_run=False,
        )
        print(result)
        return

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
