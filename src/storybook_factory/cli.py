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

    # -----------------------
    # import-orders command
    # -----------------------
    p_import = subparsers.add_parser(
        "import-orders",
        help=(
            "Read a CSV of orders and create a pending_orders/<order_id>/ folder "
            "with a brief.yaml for each row."
        ),
    )
    p_import.add_argument("--csv", required=True, help="Path to the orders CSV file")
    p_import.add_argument(
        "--theme",
        required=True,
        help="Theme name to embed in every brief (e.g. dragon_realm)",
    )
    p_import.add_argument(
        "--outfit-hint",
        default="simple medieval fantasy adventure clothing, not modern clothing",
        help="Outfit hint to embed in every brief",
    )
    p_import.add_argument(
        "--pending-dir",
        default="pending_orders",
        help="Root folder for pending order folders (default: pending_orders/)",
    )

    # -----------------------
    # process-order command
    # -----------------------
    p_proc = subparsers.add_parser(
        "process-order",
        help=(
            "Run the full pipeline for a single pending order: "
            "sort character photos → brief2json → build."
        ),
    )
    p_proc.add_argument(
        "--order-id",
        required=True,
        help="The order ID folder name inside --pending-dir (e.g. 1024_DRA)",
    )
    p_proc.add_argument(
        "--pending-dir",
        default="pending_orders",
        help="Root folder containing pending order folders (default: pending_orders/)",
    )
    p_proc.add_argument(
        "--assets-dir",
        default="assets",
        help="Base assets folder — style_reference/ is read from here (default: assets/)",
    )
    p_proc.add_argument(
        "--image-provider",
        choices=["mock", "folder", "gpt-image"],
        default="gpt-image",
        help="Image generation backend (default: gpt-image)",
    )
    p_proc.add_argument(
        "--interior-model",
        default="gpt-image-1",
        help="Model for interior pages (default: gpt-image-1)",
    )
    p_proc.add_argument(
        "--cover-model",
        default=None,
        help="Model for cover pages (default: same as --interior-model)",
    )
    p_proc.add_argument(
        "--image-quality",
        choices=["low", "medium", "high", "auto"],
        default="high",
        help="Image quality (default: high)",
    )
    p_proc.add_argument(
        "--output-format",
        choices=["print", "download"],
        default="download",
        help="Output format profile (default: download)",
    )
    p_proc.add_argument(
        "--dry-run",
        action="store_true",
        help="Print prompts without generating images",
    )

    args = parser.parse_args()

    # -----------------------
    # Handle commands
    # -----------------------
    if args.command == "import-orders":
        import csv

        import yaml as _yaml

        csv_path = Path(args.csv).resolve()
        pending_dir = Path(args.pending_dir).resolve()

        if not csv_path.exists():
            print(f"CSV not found: {csv_path}")
            sys.exit(1)

        # Expected columns (case-insensitive match)
        COL_MAP = {
            "order_id": "order_id",
            "human-01_desc": "human_01_desc",
            "human-02_desc": "human_02_desc",
            "companion_desc": "companion_desc",
            "dedication_text": "dedication_text",
        }

        created = 0
        skipped = 0

        with csv_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            # Normalise header keys: lowercase + strip whitespace
            if reader.fieldnames is None:
                print("CSV has no headers.")
                sys.exit(1)

            for raw_row in reader:
                row = {k.strip().lower(): v.strip() for k, v in raw_row.items()}

                order_id = row.get("order_id", "").strip()
                if not order_id:
                    print("  Skipping row with missing Order_ID")
                    skipped += 1
                    continue

                order_dir = pending_dir / order_id
                brief_path = order_dir / "brief.yaml"

                if brief_path.exists():
                    print(f"  [{order_id}] already exists — skipping")
                    skipped += 1
                    continue

                order_dir.mkdir(parents=True, exist_ok=True)
                (order_dir / "characters").mkdir(exist_ok=True)

                brief: dict[str, str] = {"order_id": order_id, "theme": args.theme}

                dedication = row.get("dedication_text", "").strip()
                if dedication:
                    brief["dedication_text"] = dedication

                for csv_col, yaml_key in COL_MAP.items():
                    if yaml_key in ("order_id", "dedication_text"):
                        continue
                    val = row.get(csv_col, "").strip()
                    if val:
                        brief[yaml_key] = val

                brief["outfit_hint"] = args.outfit_hint

                brief_path.write_text(
                    _yaml.dump(brief, allow_unicode=True, sort_keys=False),
                    encoding="utf-8",
                )
                print(f"  [{order_id}] created → {brief_path}")
                created += 1

        print(f"\nDone. {created} order(s) created, {skipped} skipped.")
        return

    if args.command == "process-order":
        import yaml as _yaml

        from .image_sorter import GeminiImageSorter

        pending_dir = Path(args.pending_dir).resolve()
        order_dir = pending_dir / args.order_id

        brief_path = order_dir / "brief.yaml"
        if not brief_path.exists():
            print(f"No brief found at {brief_path}")
            sys.exit(1)

        chars_root = order_dir / "characters"
        chars_root.mkdir(exist_ok=True)

        config_dir = order_dir / "config"
        output_dir = order_dir / "out"
        assets_dir = Path(args.assets_dir).resolve()
        _theme_name = (_yaml.safe_load(brief_path.read_text()) or {}).get("theme", "")
        _scene_packs_root = Path(__file__).resolve().parent.parent / "scene_packs"
        # Support both flat (scene_packs/dragon_realm.yaml) and
        # subdirectory (scene_packs/dragon_realm/dragon_realm.yaml) layouts.
        theme_path = _scene_packs_root / f"{_theme_name}.yaml"
        if not theme_path.exists():
            theme_path = _scene_packs_root / _theme_name / f"{_theme_name}.yaml"

        if not theme_path.exists():
            print(f"Scene pack not found: {theme_path}")
            sys.exit(1)

        # ── Step 1: sort character photos via Gemini ──────────────────────────
        image_extensions = (".png", ".jpg", ".jpeg", ".webp")

        _brief_raw = _yaml.safe_load(brief_path.read_text()) or {}
        char_roles: list[str] = []
        char_descs: dict[str, str] = {}
        for role in ("human_01", "human_02"):
            desc = _brief_raw.get(f"{role}_desc", "").strip()
            if desc:
                char_roles.append(role)
                char_descs[role] = desc
        companion_desc_val = _brief_raw.get("companion_desc", "").strip()
        if companion_desc_val:
            char_roles.append("companion")
            char_descs["companion"] = companion_desc_val

        ref_description_string = ""
        sorted_ref_paths: list[Path] = []

        # Check if photos are pre-organised into per-role subfolders
        # e.g. characters/human_01/*.jpg, characters/human_02/*.jpg
        # If so, skip Gemini sorting entirely — use the folder layout as ground truth.
        pre_sorted: dict[str, list[Path]] = {}
        for role in char_roles:
            role_dir = chars_root / role
            if role_dir.is_dir():
                role_photos = sorted(
                    p
                    for p in role_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in image_extensions
                )
                if role_photos:
                    pre_sorted[role] = role_photos

        if pre_sorted:
            # Build refs directly from subfolders — no Gemini call needed
            print(
                f"[{args.order_id}] Using pre-sorted character subfolders (skipping Gemini sort)."
            )
            style_ref_dir = assets_dir / "style_reference"
            style_refs = (
                sorted(
                    p
                    for p in style_ref_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in image_extensions
                )
                if style_ref_dir.exists()
                else []
            )

            sorted_ref_paths = list(style_refs)
            current_idx = 1
            from collections import defaultdict

            char_to_ids: dict[str, list[str]] = defaultdict(list)
            for _p in style_refs:
                char_to_ids["style_reference"].append(f"[{current_idx}]")
                current_idx += 1
            for role in char_roles:
                for p in pre_sorted.get(role, []):
                    sorted_ref_paths.append(p)
                    char_to_ids[role].append(f"[{current_idx}]")
                    current_idx += 1

            lines = []
            if "style_reference" in char_to_ids:
                lines.append(
                    f"style_reference: {', '.join(char_to_ids['style_reference'])}"
                )
            for role in char_roles:
                if role in char_to_ids:
                    lines.append(f"{role}: {', '.join(char_to_ids[role])}")
            ref_description_string = "REFERENCES: " + "\n".join(lines)

            for i, p in enumerate(sorted_ref_paths):
                print(f"  [{i+1}] {p.name}")
            print()
        else:
            # Fall back to flat folder + Gemini sorting
            all_refs = sorted(
                p
                for p in chars_root.iterdir()
                if p.is_file() and p.suffix.lower() in image_extensions
            )

            if all_refs and char_roles:
                print(
                    f"[{args.order_id}] Sorting {len(all_refs)} photo(s) with Gemini..."
                )
                try:
                    sorter = GeminiImageSorter()
                    ref_description_string, sorted_ref_paths = (
                        sorter.get_reference_mapping(
                            all_refs,
                            char_roles,
                            style_ref_dir=assets_dir / "style_reference",
                            character_descs=char_descs,
                        )
                    )
                    for i, p in enumerate(sorted_ref_paths):
                        print(f"  [{i+1}] {p.name}")
                    print()
                except Exception as e:
                    print(
                        f"  Warning: Gemini sorter failed: {e}. Continuing without refs."
                    )
                    ref_description_string = ""
                    sorted_ref_paths = []
            else:
                if not all_refs:
                    print(
                        f"[{args.order_id}] No photos in {chars_root} — generating without character refs."
                    )

        # ── Step 2: brief2json ────────────────────────────────────────────────
        print(f"[{args.order_id}] Generating page prompts...")
        generate_from_brief(
            brief_path,
            theme_path,
            config_dir,
            ref_description_string=ref_description_string,
        )

        # ── Step 3: build ─────────────────────────────────────────────────────
        print(f"[{args.order_id}] Building images...")
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
            sorted_ref_paths=sorted_ref_paths if sorted_ref_paths else None,
        )
        print(result)
        return

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
        _scene_packs_root = Path(__file__).resolve().parent.parent / "scene_packs"
        # Support both flat (scene_packs/dragon_realm.yaml) and
        # subdirectory (scene_packs/dragon_realm/dragon_realm.yaml) layouts.
        theme_path = _scene_packs_root / f"{theme_name}.yaml"
        if not theme_path.exists():
            theme_path = _scene_packs_root / theme_name / f"{theme_name}.yaml"

        if not theme_path.exists():
            print(f"Scene pack not found: {theme_path}")
            sys.exit(1)

        out_dir = Path(args.out)
        # Use assets_dir from args if available, otherwise default to "assets"
        assets_dir = Path(getattr(args, "assets_dir", "assets")).resolve()

        # Build ref description string by running Gemini sorter on assets/characters/
        ref_description_string = ""

        # Derive role labels and descriptions directly from the brief's desc keys.
        # No real names ever — roles are always human_01, human_02, companion.
        import yaml as _yaml

        _brief_raw = _yaml.safe_load(brief_path.read_text()) or {}
        char_roles: list[str] = []
        char_descs: dict[str, str] = {}
        for role in ("human_01", "human_02"):
            desc = _brief_raw.get(f"{role}_desc", "").strip()
            if desc:
                char_roles.append(role)
                char_descs[role] = desc
        companion_desc = _brief_raw.get("companion_desc", "").strip()
        if companion_desc:
            char_roles.append("companion")
            char_descs["companion"] = companion_desc

        image_extensions = (".png", ".jpg", ".jpeg", ".webp")
        chars_root = assets_dir / "characters"

        # Collect all images directly inside assets/characters/ (top-level only)
        all_refs: list[Path] = []
        if chars_root.exists():
            all_refs = sorted(
                p
                for p in chars_root.iterdir()
                if p.is_file() and p.suffix.lower() in image_extensions
            )

        if all_refs and char_roles:
            from .image_sorter import GeminiImageSorter

            try:
                sorter = GeminiImageSorter()
                ref_description_string, sorted_paths = sorter.get_reference_mapping(
                    all_refs,
                    char_roles,
                    style_ref_dir=assets_dir / "style_reference",
                    character_descs=char_descs,
                )
                if sorted_paths:
                    for i, p in enumerate(sorted_paths):
                        print(f"[{i+1}] {p}")
                    print("---------------------------------------\n")
            except Exception as e:
                print(
                    f"Warning: Gemini sorter failed: {e}. Falling back to no references."
                )
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
