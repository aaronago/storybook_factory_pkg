# overlays.py (new)
from __future__ import annotations

import textwrap
from dataclasses import dataclass

from PIL import Image, ImageDraw, ImageFont


@dataclass
class TextStyle:
    font_path: str | None
    size: int
    fill: str = "black"
    stroke_width: int = 0
    stroke_fill: str = "white"
    line_spacing: float = 1.15


def _load_font(style: TextStyle) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    if style.font_path:
        return ImageFont.truetype(style.font_path, style.size)
    return ImageFont.load_default()


def _measure(
    draw: ImageDraw.ImageDraw, font: ImageFont.ImageFont, text: str
) -> tuple[int, int]:
    bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=0)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def _wrap_to_width(
    draw: ImageDraw.ImageDraw, font: ImageFont.ImageFont, text: str, max_w: int
) -> str:
    # greedy wrap by words
    words = text.split()
    lines = []
    cur = []
    for w in words:
        trial = (" ".join(cur + [w])).strip()
        tw, _ = _measure(draw, font, trial)
        if tw <= max_w or not cur:
            cur.append(w)
        else:
            lines.append(" ".join(cur))
            cur = [w]
    if cur:
        lines.append(" ".join(cur))
    return "\n".join(lines)


def draw_text_in_region(
    img: Image.Image,
    *,
    text: str,
    region_px: tuple[int, int, int, int],  # x,y,w,h
    style: TextStyle,
    align: str = "center",
    min_size: int = 18,
) -> Image.Image:
    # Work in RGB so stroke behaves consistently, then convert back if needed
    rgb = img.convert("RGB")
    draw = ImageDraw.Draw(rgb)

    x, y, w, h = region_px

    # shrink-to-fit loop
    size = style.size
    while size >= min_size:
        font = (
            ImageFont.truetype(style.font_path, size)
            if style.font_path
            else ImageFont.load_default()
        )
        wrapped = _wrap_to_width(draw, font, text, w)
        tw, th = _measure(draw, font, wrapped)

        # allow a bit of spacing
        if tw <= w and th <= h:
            # position
            tx = x + (w - tw) // 2 if align == "center" else x
            ty = y + (h - th) // 2

            draw.multiline_text(
                (tx, ty),
                wrapped,
                font=font,
                fill=style.fill,
                align=align,
                spacing=int(size * (style.line_spacing - 1.0)),
                stroke_width=style.stroke_width,
                stroke_fill=style.stroke_fill,
            )
            return rgb

        size -= 2

    # fallback: draw smallest
    font = (
        ImageFont.truetype(style.font_path, min_size)
        if style.font_path
        else ImageFont.load_default()
    )
    wrapped = _wrap_to_width(draw, font, text, w)
    tw, th = _measure(draw, font, wrapped)
    tx = x + (w - tw) // 2
    ty = y + (h - th) // 2
    draw.multiline_text((tx, ty), wrapped, font=font, fill=style.fill, align="center")
    return rgb
    draw.multiline_text((tx, ty), wrapped, font=font, fill=style.fill, align="center")
    return rgb
