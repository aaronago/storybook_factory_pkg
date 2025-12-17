from __future__ import annotations

import base64
import shutil
from collections.abc import Iterable
from pathlib import Path

from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont

# Simple default font for placeholder text
DEFAULT_FONT = ImageFont.load_default()


def ensure_dir(p: Path):
    """Create directory (and parents) if it doesn’t exist."""
    p.mkdir(parents=True, exist_ok=True)


class ImageProvider:
    """
    Handles creation of page/cover PNGs for the pipeline.

    mode:
      - "mock"
          Draws simple placeholder boxes with the prompt text inside.
      - "folder"
          Copies existing PNGs from assets_dir into the output directory.
      - "gpt-image"
          Uses OpenAI's image API to generate the PNGs directly from prompts.
    """

    def __init__(
        self,
        out_dir: Path,
        interior_px,
        cover_px,
        mode: str = "mock",
        openai_cover_model: str | None = None,
        openai_interior_model: str | None = None,
        assets_dir: Path | None = None,
        dry_run: bool = False,
    ):
        self.out_dir = out_dir
        self.interior_px = self._normalize_px(interior_px, name="interior_px")
        self.cover_px = self._normalize_px(cover_px, name="cover_px")
        self.mode = mode
        self.cover_model = openai_cover_model
        self.interior_model = openai_interior_model
        self.assets_dir = assets_dir
        self.dry_run = dry_run

        if self.mode == "gpt-image":
            self.client = OpenAI()
        else:
            self.client = None

    @staticmethod
    def _normalize_px(val, name: str):
        # Accept (w,h), [w,h], {"w":..,"h":..}
        if isinstance(val, dict):
            w = int(val.get("w"))
            h = int(val.get("h"))
            return (w, h)
        if isinstance(val, (list, tuple)) and len(val) == 2:
            return (int(val[0]), int(val[1]))
        raise TypeError(
            f"{name} must be tuple/list (w,h) or dict {{'w':..,'h':..}}, got: {type(val)} -> {val!r}"
        )

    @staticmethod
    def _trim_prompt(prompt: str, limit: int) -> str:
        """
        Trim a prompt to <= limit chars while preserving:
        - the beginning (style/bible)
        - the end (the unique scene content)
        """
        p = prompt.strip()
        if len(p) <= limit:
            return p

        # Keep slightly more of the front than the tail to preserve bible cues
        head_budget = int(limit * 0.6)
        tail_budget = limit - head_budget - 12  # room for separator

        head = p[:head_budget]
        # Avoid chopping mid-line if possible
        if "\n" in head:
            head = head.rsplit("\n", 1)[0]

        tail = p[-tail_budget:]
        # Avoid starting tail mid-word if possible
        if "\n" in tail:
            tail = tail.split("\n", 1)[-1]

        trimmed = head + "\n...\n" + tail
        return trimmed[:limit].strip()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _placeholder(self, filename: str, text: str, cover: bool = False) -> Path:
        """
        Draw a simple placeholder image with a border and text so you
        can see layout & prompts without paying for image generation.
        """
        w, h = self.cover_px if cover else self.interior_px
        img = Image.new("RGB", (w, h), "white")
        draw = ImageDraw.Draw(img)

        # Frame
        draw.rectangle((10, 10, w - 10, h - 10), outline="black", width=5)

        # Very simple word-wrapping
        words = text.split()
        lines: list[str] = []
        line = ""
        for word in words:
            candidate = (line + " " + word).strip()
            if len(candidate) > 60:
                lines.append(line)
                line = word
            else:
                line = candidate
        if line:
            lines.append(line)

        # Draw text near the top
        y = 30
        for ln in lines[:80]:
            draw.text((30, y), ln, fill="black", font=DEFAULT_FONT)
            y += 20

        out_path = self.out_dir / filename
        img.save(out_path, "PNG")
        return out_path

    def _copy_from_assets(self, filename: str) -> Path | None:
        """
        Try to copy an existing PNG from assets_dir to out_dir.
        Returns the destination path if found, else None.
        """
        if not self.assets_dir:
            return None
        src = self.assets_dir / filename
        if src.exists():
            dst = self.out_dir / filename
            shutil.copy2(src, dst)
            return dst
        return None

    def _generate_with_openai(
        self,
        filename: str,
        prompt: str,
        cover: bool = False,
        reference_image: Path | None = None,
    ) -> Path:
        """
        Call OpenAI's image API to generate a PNG for this page or cover,
        then resize it to the exact pixel size required by the pipeline.

        If reference_image is provided, use images.edits() to anchor style.
        """
        if self.client is None:
            raise RuntimeError(
                "OpenAI client not initialized; make sure mode='gpt-image' "
                "and OPENAI_API_KEY is set."
            )

        if cover:
            api_size = "1536x1024"  # landscape-ish
            target_w, target_h = self.cover_px
            model_name = self.cover_model
        else:
            api_size = "1024x1536"  # portrait
            target_w, target_h = self.interior_px
            model_name = self.interior_model

        if not model_name:
            raise RuntimeError("No OpenAI model configured for this image type.")

        # DALL·E 2 has a strict prompt length limit (~1000 chars).
        if (not cover) and (model_name == "dall-e-2"):
            prompt = self._trim_prompt(prompt, 1000)

        if self.dry_run:
            print(
                f"[dry-run] would generate {filename} using model={model_name} size={api_size}"
            )
            if reference_image:
                print(f"[dry-run] reference_image={reference_image}")
            print(prompt)
            print("-" * 80)
            return self._placeholder(filename, f"[DRY RUN] {prompt}", cover=cover)

        # --- IMPORTANT: branch here ---
        if reference_image:
            with open(reference_image, "rb") as f:
                kwargs = dict(
                    model=model_name,  # "gpt-image-1"
                    image=[f],  # NOTE: list of file objects
                    prompt=prompt,
                    size=api_size,
                )
                # Optional: only if your SDK supports it
                kwargs["input_fidelity"] = "high"

                result = self.client.images.edit(**kwargs)

        else:
            result = self.client.images.generate(
                model=model_name,
                prompt=prompt,
                size=api_size,
                n=1,
            )

        b64_data = result.data[0].b64_json
        img_bytes = base64.b64decode(b64_data)

        out_path = self.out_dir / filename
        with open(out_path, "wb") as f:
            f.write(img_bytes)

        img = Image.open(out_path).convert("RGB")
        img = img.resize((target_w, target_h), Image.LANCZOS)
        img.save(out_path, "PNG")
        return out_path

    # ------------------------------------------------------------------
    # Public API used by the pipeline
    # ------------------------------------------------------------------

    def render_interior(self, interior_prompts: Iterable[dict]) -> None:
        """
        Generate/copy all interior page images according to mode.
        """
        for p in interior_prompts:
            fname = p["file"]
            prompt = p["prompt"]

            if self.mode == "folder":
                got = self._copy_from_assets(fname)
                if got is not None:
                    continue
                self._placeholder(fname, f"[MISSING FILE] {prompt}", cover=False)

            elif self.mode == "gpt-image":
                out_path = self.out_dir / fname
                if out_path.exists():
                    print(f"[gpt-image] skipping existing interior image: {fname}")
                    continue
                print(f"[gpt-image] generating interior image: {fname}")
                self._generate_with_openai(fname, prompt, cover=False)

            else:
                self._placeholder(fname, f"[INTERIOR PAGE] {prompt}", cover=False)

    def render_covers(self, covers: dict[str, dict]) -> None:
        """
        Generate/copy cover images (front & back).

        Supports optional style reference images:
          - cov["style_reference_image"] : str path to an image file
        """
        for key, cov in covers.items():
            fname = cov.get("file")
            prompt = cov.get("prompt", "")
            if not fname:
                continue

            # Optional reference image path (relative to project root or CWD)
            ref = cov.get("style_reference_image")
            ref_path = Path(ref) if ref else None

            # If the path is relative, resolve it to an absolute path
            # using the current working directory as the base.
            # (If your project root differs from CWD, replace Path.cwd() accordingly.)
            if ref_path and not ref_path.is_absolute():
                ref_path = (Path.cwd() / ref_path).resolve()

            # Validate reference file exists; if not, fall back to None (and print warning)
            if ref_path and not ref_path.exists():
                print(f"[warn] cover {key} reference image not found: {ref_path}")
                ref_path = None

            if self.mode == "folder":
                got = self._copy_from_assets(fname)
                if got is not None:
                    continue
                self._placeholder(fname, f"[MISSING COVER {key}] {prompt}", cover=True)

            elif self.mode == "gpt-image":
                out_path = self.out_dir / fname
                if out_path.exists():
                    print(f"[gpt-image] skipping existing cover image: {fname}")
                    continue

                if ref_path:
                    print(
                        f"[gpt-image] generating cover image (with reference): {fname}"
                    )
                    print(f"[gpt-image]   reference: {ref_path}")
                else:
                    print(f"[gpt-image] generating cover image: {fname}")

                self._generate_with_openai(
                    fname,
                    prompt,
                    cover=True,
                    reference_image=ref_path,
                )

            else:
                self._placeholder(fname, f"[COVER {key}] {prompt}", cover=True)
