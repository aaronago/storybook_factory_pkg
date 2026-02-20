# character_sheets.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RefSubject:
    id: str
    display_name: str
    kind: str  # "child" or "pet"
    extra: str  # age/breed/etc (optional)
    photos: list[Path]


def parse_character_ids(page_prompts: dict[str, Any]) -> list[str]:
    """
    Optional (NO paths). If page_prompts.json contains:
      { "characters": ["id1", "id2"] }
    return those IDs so the CLI can filter discovered subjects.

    If missing, returns [] meaning "no filtering".
    """
    ids = page_prompts.get("characters")
    if not ids:
        return []
    return [str(x).strip().lower() for x in ids if str(x).strip()]


def _safe_name(s: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "-" for ch in s).strip("-")


def _infer_kind(subject_dir: Path) -> str:
    """
    Best-effort inference:
      - kind.txt containing "pet" or "child" wins
      - folder name containing "pet" => pet
      - otherwise default to child
    """
    kind_file = subject_dir / "kind.txt"
    if kind_file.exists():
        val = kind_file.read_text().strip().lower()
        if "pet" in val:
            return "pet"
        if "child" in val:
            return "child"
    if "pet" in subject_dir.name.lower():
        return "pet"
    return "child"


def discover_character_refs_from_assets(assets_dir: Path) -> list[RefSubject]:
    """
    Looks for:
      assets_dir/characters/<subject_id>/reference/*.(png|jpg|jpeg|webp)
    If reference/ doesn't exist, it will use any images under the subject folder.

    This avoids storing paths inside page_prompts.json.
    """
    characters_root = assets_dir / "characters"
    if not characters_root.exists():
        return []

    subjects: list[RefSubject] = []
    for subject_dir in sorted([p for p in characters_root.iterdir() if p.is_dir()]):
        kind = _infer_kind(subject_dir)

        ref_dir = subject_dir / "reference"
        search_dir = ref_dir if ref_dir.exists() else subject_dir

        photos = []
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
            photos.extend(sorted(search_dir.glob(ext)))

        if not photos:
            continue

        sid = subject_dir.name
        subjects.append(
            RefSubject(
                id=sid,
                display_name=sid,
                kind=kind,
                extra="",
                photos=photos,
            )
        )

    return subjects


def _make_subject_prompt(sub: RefSubject) -> str:
    """
    This is tuned for your current win condition:
    soft hand-drawn coloring-book line art (not chibi, not watercolor).
    """
    identity = (
        "Use the provided photo(s) ONLY to preserve the subject's recognizable identity "
        "(face shape, hairstyle, proportions, distinctive features)."
    )

    if sub.kind == "pet":
        details = (
            "Full-body character sheet of the SAME pet from the photos. "
            "Accurate breed features, ears, muzzle shape, and body proportions."
        )
    else:
        details = (
            "Full-body character sheet of the SAME child from the photos. "
            "Natural child proportions (NOT chibi): realistic head-to-body ratio, "
            "defined limbs, visible fingers, proportional feet."
        )

    style = (
        "Style: coloring-book line art. Soft, hand-drawn ink lines. "
        "Clean outlines ONLY. White background.\n"
        "Strict rules:\n"
        "- NO shading, NO grayscale, NO gradients\n"
        "- NO cross-hatching, NO heavy texture fills\n"
        "- keep interior detail minimal and colorable\n"
        "- single subject only (no extra people/pets)\n"
        "- minimal ground line only"
    )

    return f"{identity}\n{details}\n{style}"


def make_reference_sheets(
    provider: Any,  # ImageProvider
    subjects: list[RefSubject],
    refs_dir: Path,
    force: bool = False,
) -> dict[str, Path]:
    """
    Generates one reference sheet per subject into:
      <images_dir>/refs/<kind>-<id>.png

    Returns a map like:
      {"child:colette": Path(...), "pet:winnie": Path(...)}
    """
    refs_dir.mkdir(parents=True, exist_ok=True)
    out_map: dict[str, Path] = {}

    for sub in subjects:
        out_name = f"refs/{sub.kind}-{_safe_name(sub.id)}.png"
        out_path = refs_dir / Path(out_name).name
        out_map[f"{sub.kind}:{sub.id}"] = out_path

        if out_path.exists() and not force:
            continue

        if not sub.photos:
            raise FileNotFoundError(f"No photos provided for {sub.kind}:{sub.id}")

        photo = sub.photos[0]
        if not photo.exists():
            raise FileNotFoundError(f"Missing photo: {photo}")

        prompt = _make_subject_prompt(sub)

        # Transform photo -> line art
        provider.apply_edit(
            input_images=[photo],
            prompt=prompt,
            out_filename=out_name,  # writes under images_dir/refs/...
            cover=False,
        )

    return out_map
