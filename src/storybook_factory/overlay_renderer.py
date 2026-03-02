from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFilter, ImageFont


@dataclass(frozen=True)
class TextStyle:
    """
    Text overlay style.

    Notes:
    - font_path may be None to use PIL default font.
    - line_spacing is a multiplier (1.0 = tight, 1.15 = a bit looser).
    - glow_* enables a soft outer-glow behind the text for better integration
      on illustrated covers.
    """

    font_path: str | None
    size: int
    fill: str = "black"
    stroke_width: int = 0
    stroke_fill: str = "white"
    line_spacing: float = 1.15  # multiplier

    # --- NEW: glow support ---
    glow_radius: int = 0  # pixels; 0 disables
    glow_fill: str = "#FFB347"  # warm gold by default
    glow_alpha: int = 160  # 0..255
    glow_stroke_width: int | None = None  # if None, use stroke_width
    glow_stroke_fill: str | None = None  # if None, use glow_fill


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


def _clamp_alpha(a: int) -> int:
    return max(0, min(255, int(a)))


def _draw_glow(
    base_rgb: Image.Image,
    *,
    text: str,
    xy: tuple[int, int],
    font: ImageFont.ImageFont,
    spacing: int,
    align: str,
    style: TextStyle,
) -> Image.Image:
    """
    Draw a blurred glow behind text and composite onto base image.
    Returns an RGB image.
    """
    if style.glow_radius <= 0 or style.glow_alpha <= 0:
        return base_rgb

    # Build an RGBA layer for glow
    glow_layer = Image.new("RGBA", base_rgb.size, (0, 0, 0, 0))
    glow_draw = ImageDraw.Draw(glow_layer)

    glow_sw = style.glow_stroke_width
    if glow_sw is None:
        glow_sw = style.stroke_width

    glow_sf = style.glow_stroke_fill
    if glow_sf is None:
        glow_sf = style.glow_fill

    # Draw text in glow color first
    glow_draw.multiline_text(
        xy,
        text,
        font=font,
        fill=style.glow_fill,
        spacing=spacing,
        align=align,
        stroke_width=int(glow_sw),
        stroke_fill=glow_sf,
    )

    # Blur the glow
    glow_layer = glow_layer.filter(ImageFilter.GaussianBlur(radius=style.glow_radius))

    # Apply global alpha to the glow layer
    a = _clamp_alpha(style.glow_alpha)
    if a < 255:
        # Multiply existing alpha by a/255
        r, g, b, alpha = glow_layer.split()
        alpha = alpha.point(lambda p: int(p * (a / 255.0)))
        glow_layer = Image.merge("RGBA", (r, g, b, alpha))

    # Composite glow under the base image
    base_rgba = base_rgb.convert("RGBA")
    composited = Image.alpha_composite(base_rgba, glow_layer)
    return composited.convert("RGB")


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

    Glow behavior:
      - If style.glow_radius > 0 and style.glow_alpha > 0, a soft blurred
        duplicate of the wrapped text is drawn behind the final text.
    """
    out = img.convert("RGB")
    draw = ImageDraw.Draw(out)

    for ov in overlays:
        if ov.get("type") != "text":

            continue

        text = (ov.get("content") or "").strip()
        if not text:
            continue
        print("Rendering overlay:", text)

        region_pct = ov["region_pct"]
        style_key = ov["style"]

        # Skip if style not found
        if style_key not in styles:
            print(f"[overlay] WARNING: style '{style_key}' not found; skipping overlay")
            continue

        style = styles[style_key]

        x, y, w, h = _pct_region_to_px(out, region_pct)

        # shrink-to-fit loop
        size = int(style.size)
        while size >= min_font_size:
            font = _load_font(style, size)

            # Existing behavior: spacing is computed as extra pixels per line
            spacing = int(size * (style.line_spacing - 1.0))

            wrapped = _wrap_to_width(draw, text, font, max_w=w, spacing=spacing)
            tw, th = _multiline_bbox(draw, wrapped, font, spacing)

            if tw <= w and th <= h:
                tx = x + (w - tw) // 2
                ty = y + (h - th) // 2
                xy = (tx, ty)

                # Draw glow first (behind)
                out = _draw_glow(
                    out,
                    text=wrapped,
                    xy=xy,
                    font=font,
                    spacing=spacing,
                    align="center",
                    style=style,
                )
                draw = ImageDraw.Draw(out)

                # Draw actual text on top
                draw.multiline_text(
                    xy,
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
