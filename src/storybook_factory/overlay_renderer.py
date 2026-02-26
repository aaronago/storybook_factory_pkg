from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class TextStyle:
    font_path: str | None
    size: int
    fill: str = "black"
    stroke_width: int = 0
    stroke_fill: str = "white"
    line_spacing: float = 1.15  # multiplier


def _load_font(
    style: TextStyle, size: int
) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    if not style.font_path:
        return ImageFont.load_default()

    font_path = Path(style.font_path)
    if not font_path.is_absolute():
        # Resolve relative to project root (two levels up from this file)
        project_root = Path(__file__).resolve().parents[2]
        font_path = project_root / font_path
    return ImageFont.truetype(str(font_path), size)


def _multiline_bbox(
    draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, spacing: int
) -> tuple[int, int]:
    bbox = draw.multiline_textbbox(
        (0, 0), text, font=font, spacing=spacing, align="center"
    )
    return (bbox[2] - bbox[0], bbox[3] - bbox[1])


def _wrap_to_width(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    max_w: int,
    spacing: int,
) -> str:
    words = text.split()
    if not words:
        return ""

    lines: list[str] = []
    cur: list[str] = []

    for w in words:
        trial = " ".join(cur + [w])
        tw, _ = _multiline_bbox(draw, trial, font, spacing)
        if tw <= max_w or not cur:
            cur.append(w)
        else:
            lines.append(" ".join(cur))
            cur = [w]

    if cur:
        lines.append(" ".join(cur))

    return "\n".join(lines)


def _pct_region_to_px(
    img: Image.Image, region_pct: dict[str, float]
) -> tuple[int, int, int, int]:
    W, H = img.size
    x = int(W * (region_pct["x"] / 100.0))
    y = int(H * (region_pct["y"] / 100.0))
    w = int(W * (region_pct["w"] / 100.0))
    h = int(H * (region_pct["h"] / 100.0))
    return x, y, w, h


def apply_overlays(
    img: Image.Image,
    overlays: list[dict[str, Any]],
    styles: dict[str, TextStyle],
    *,
    min_font_size: int = 18,
) -> Image.Image:
    """
    Applies overlays onto img and returns a NEW image.
    Expects each overlay:
      - type: "text"
      - content: string
      - region_pct: {x,y,w,h} percentages
      - style: style key in styles dict
    """
    out = img.convert("RGB")
    draw = ImageDraw.Draw(out)

    for ov in overlays:
        if ov.get("type") != "text":
            continue

        text = (ov.get("content") or "").strip()
        if not text:
            continue

        region_pct = ov["region_pct"]
        style_key = ov["style"]

        # Skip if style not found
        if style_key not in styles:
            print(f"[overlay] WARNING: style '{style_key}' not found; skipping overlay")
            continue

        style = styles[style_key]

        x, y, w, h = _pct_region_to_px(out, region_pct)

        # shrink-to-fit loop
        size = style.size
        while size >= min_font_size:
            font = _load_font(style, size)
            spacing = int(size * (style.line_spacing - 1.0))
            wrapped = _wrap_to_width(draw, text, font, max_w=w, spacing=spacing)
            tw, th = _multiline_bbox(draw, wrapped, font, spacing)

            if tw <= w and th <= h:
                tx = x + (w - tw) // 2
                ty = y + (h - th) // 2
                draw.multiline_text(
                    (tx, ty),
                    wrapped,
                    font=font,
                    fill=style.fill,
                    spacing=spacing,
                    align="center",
                    stroke_width=style.stroke_width,
                    stroke_fill=style.stroke_fill,
                )
                break

            size -= 2

    return out
    return out
