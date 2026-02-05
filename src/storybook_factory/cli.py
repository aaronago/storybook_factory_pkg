import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

from .generator import generate_from_brief
from .pipeline import run_pipeline
from .settings import settings

load_dotenv()


print(settings.summary())


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
    # build command
    # -----------------------
    p2 = subparsers.add_parser(
        "build",
        help="Generate all images, PDFs, and final ZIP package",
    )
    p2.add_argument("--config-dir", required=True, help="Config directory (JSON files)")
    p2.add_argument(
        "--output-dir", required=True, help="Output directory for artifacts"
    )
    p2.add_argument(
        "--assets-dir",
        default="assets",
        help="Folder with any static assets (if used by pipeline)",
    )
    p2.add_argument(
        "--image-provider",
        choices=["mock", "folder", "gpt-image"],
        default="mock",
        help="Image generation backend",
    )
    # NEW: separate models for covers vs interiors
    p2.add_argument(
        "--openai-cover-model",
        default="gpt-image-1",
        help="OpenAI model for cover images (default: gpt-image-1)",
    )
    p2.add_argument(
        "--openai-interior-model",
        default="dall-e-2",
        help="OpenAI model for interior coloring pages (default: dall-e-2)",
    )
    p2.add_argument(
        "--dry-run",
        action="store_true",
        help="Print prompts without generating images",
    )

    # -----------------------
    # covers-only command
    # -----------------------
    p3 = subparsers.add_parser(
        "covers",
        help="Generate ONLY front/back cover images (skip interiors)",
    )
    p3.add_argument("--config-dir", required=True, help="Config directory (JSON files)")
    p3.add_argument("--output-dir", required=True, help="Output directory for covers")
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
        "--openai-cover-model",
        default="gpt-image-1",
        help="OpenAI model for cover images (default: gpt-image-1)",
    )
    p3.add_argument(
        "--openai-interior-model",
        default="dall-e-2",
        help="OpenAI model for interior coloring pages (not used in covers-only mode)",
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

    elif args.command == "build":
        result = run_pipeline(
            config_dir=Path(args.config_dir),
            output_dir=Path(args.output_dir),
            assets_dir=Path(args.assets_dir),
            image_provider_mode=args.image_provider,
            # NEW: separate models
            openai_cover_model=args.openai_cover_model,
            openai_interior_model=args.openai_interior_model,
            dry_run=args.dry_run,
            covers_only=False,
            interiors_only=False,
        )
        print(result)
        return

    elif args.command == "covers":
        result = run_pipeline(
            config_dir=Path(args.config_dir),
            output_dir=Path(args.output_dir),
            assets_dir=Path(args.assets_dir),
            image_provider_mode=args.image_provider,
            openai_cover_model=args.openai_cover_model,
            openai_interior_model=args.openai_interior_model,
            dry_run=False,
            covers_only=True,
            interiors_only=False,
        )
        print(result)
        return

    else:
        parser.print_help()
