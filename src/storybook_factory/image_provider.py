# image_provider.py
from __future__ import annotations

import base64
import os
import shutil
from collections.abc import Iterable
from pathlib import Path

from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont

DEFAULT_FONT = ImageFont.load_default()


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


class ImageProvider:
    """
    Handles creation of page/cover PNGs for the pipeline.

    mode:
      - "mock"      placeholder images with prompt text
      - "folder"    copies existing PNGs from assets_dir
      - "gpt-image" generates via OpenAI Images API
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

        # Allow overrides via env
        self.interior_api_size = os.getenv("STORYBOOK_INTERIOR_API_SIZE", "1024x1536")
        self.cover_api_size = os.getenv("STORYBOOK_COVER_API_SIZE", "1536x1024")
        self.interior_quality = os.getenv("STORYBOOK_INTERIOR_QUALITY", "high")
        self.cover_quality = os.getenv("STORYBOOK_COVER_QUALITY", "high")

    @staticmethod
    def _normalize_px(val, name: str):
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
        p = prompt.strip()
        if len(p) <= limit:
            return p
        head_budget = int(limit * 0.6)
        tail_budget = limit - head_budget - 12
        head = p[:head_budget]
        if "\n" in head:
            head = head.rsplit("\n", 1)[0]
        tail = p[-tail_budget:]
        if "\n" in tail:
            tail = tail.split("\n", 1)[-1]
        trimmed = head + "\n...\n" + tail
        return trimmed[:limit].strip()

    def _placeholder(self, filename: str, text: str, cover: bool = False) -> Path:
        w, h = self.cover_px if cover else self.interior_px
        img = Image.new("RGB", (w, h), "white")
        draw = ImageDraw.Draw(img)
        draw.rectangle((10, 10, w - 10, h - 10), outline="black", width=5)

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

        y = 30
        for ln in lines[:80]:
            draw.text((30, y), ln, fill="black", font=DEFAULT_FONT)
            y += 20

        out_path = self.out_dir / filename
        img.save(out_path, "PNG")
        return out_path

    def _copy_from_assets(self, filename: str) -> Path | None:
        if not self.assets_dir:
            return None
        src = self.assets_dir / filename
        if src.exists():
            dst = self.out_dir / filename
            shutil.copy2(src, dst)
            return dst
        return None

    # ----------------------------
    # New: candidate generation
    # ----------------------------

    def generate_candidates(
        self,
        *,
        base_filename: str,
        prompt: str,
        cover: bool = False,
        n: int = 1,
    ) -> list[Path]:
        """
        Generate N candidate images for a given prompt and return their paths.
        Does not pick the best; that belongs in the pipeline/reviewer.
        """
        if self.mode != "gpt-image":
            # For mock/folder mode, just create one placeholder candidate.
            fname = base_filename
            return [self._placeholder(fname, prompt, cover=cover)]

        if self.client is None:
            raise RuntimeError(
                "OpenAI client not initialized; mode must be 'gpt-image'."
            )

        if cover:
            api_size = self.cover_api_size
            target_w, target_h = self.cover_px
            model_name = self.cover_model
            quality = self.cover_quality
        else:
            api_size = self.interior_api_size
            target_w, target_h = self.interior_px
            model_name = self.interior_model
            quality = self.interior_quality

        if not model_name:
            raise RuntimeError("No OpenAI model configured for this image type.")

        # DALL·E 2 strict prompt limit
        if model_name == "dall-e-2":
            prompt = self._trim_prompt(prompt, 1000)

        if self.dry_run:
            print(
                f"[dry-run] would generate {n} candidates for {base_filename} model={model_name} size={api_size}"
            )
            print(prompt)
            print("-" * 80)
            return [
                self._placeholder(base_filename, f"[DRY RUN] {prompt}", cover=cover)
            ]

        # Generate N
        result = self.client.images.generate(
            model=model_name,
            prompt=prompt,
            size=api_size,
            n=n,
            quality=quality,
            output_format="png",
        )

        out_paths: list[Path] = []
        for i, data in enumerate(result.data):
            b64_data = data.b64_json
            img_bytes = base64.b64decode(b64_data)

            cand_name = self._candidate_name(base_filename, i)
            out_path = self.out_dir / cand_name
            out_path.write_bytes(img_bytes)

            img = (
                Image.open(out_path)
                .convert("RGB")
                .resize((target_w, target_h), Image.LANCZOS)
            )
            img.save(out_path, "PNG")

            out_paths.append(out_path)

        return out_paths

    def apply_edit(
        self,
        *,
        input_image: Path,
        prompt: str,
        out_filename: str,
        cover: bool = False,
    ) -> Path:
        """
        Optional edit pass for a selected candidate.
        Useful to remove shading, simplify background, thicken outlines, etc.
        """
        if self.mode != "gpt-image":
            return input_image

        if self.client is None:
            raise RuntimeError(
                "OpenAI client not initialized; mode must be 'gpt-image'."
            )

        if cover:
            api_size = self.cover_api_size
            target_w, target_h = self.cover_px
            model_name = self.cover_model
        else:
            api_size = self.interior_api_size
            target_w, target_h = self.interior_px
            model_name = self.interior_model

        if not model_name:
            raise RuntimeError("No OpenAI model configured for this image type.")

        if self.dry_run:
            print(
                f"[dry-run] would edit {input_image} into {out_filename} using model={model_name} size={api_size}"
            )
            print(prompt)
            print("-" * 80)
            return self._placeholder(
                out_filename, f"[DRY RUN EDIT] {prompt}", cover=cover
            )

        with open(input_image, "rb") as f:
            kwargs = dict(
                model=model_name,
                image=[f],
                prompt=prompt,
                size=api_size,
            )
            # Some SDKs accept this; harmless if ignored.
            kwargs["input_fidelity"] = "high"
            result = self.client.images.edit(**kwargs)

        b64_data = result.data[0].b64_json
        img_bytes = base64.b64decode(b64_data)

        out_path = self.out_dir / out_filename
        out_path.write_bytes(img_bytes)

        img = (
            Image.open(out_path)
            .convert("RGB")
            .resize((target_w, target_h), Image.LANCZOS)
        )
        img.save(out_path, "PNG")
        return out_path

    def finalize_candidate(self, *, candidate_path: Path, final_filename: str) -> Path:
        """
        Copy/rename the chosen candidate image to the final filename the rest of the pipeline expects.
        """
        final_path = self.out_dir / final_filename
        shutil.copy2(candidate_path, final_path)
        return final_path

    def cleanup_candidates(self, *, base_filename: str) -> None:
        """
        Remove candidate files for a given base filename.
        """
        stem = Path(base_filename).stem
        for p in self.out_dir.glob(f"{stem}__cand*.png"):
            try:
                p.unlink()
            except Exception:
                pass

    def _candidate_name(self, base_filename: str, idx: int) -> str:
        p = Path(base_filename)
        return f"{p.stem}__cand{idx:02d}.png"

    # ----------------------------
    # Existing public API
    # ----------------------------

    def render_interior(self, interior_prompts: Iterable[dict]) -> None:
        """
        Legacy behavior: generate one image per prompt (or placeholder).
        Kept for backwards compatibility, but the new pipeline should use
        generate_candidates + reviewer selection.
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
                print(f"[gpt-image] generating interior image (legacy): {fname}")
                cands = self.generate_candidates(
                    base_filename=fname, prompt=prompt, cover=False, n=1
                )
                self.finalize_candidate(candidate_path=cands[0], final_filename=fname)
                self.cleanup_candidates(base_filename=fname)

            else:
                self._placeholder(fname, f"[INTERIOR PAGE] {prompt}", cover=False)

    def render_covers(self, covers: dict[str, dict]) -> None:
        """
        Legacy behavior for covers.
        """
        for key, cov in covers.items():
            fname = cov.get("file")
            prompt = cov.get("prompt", "")
            if not fname:
                continue

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
                print(f"[gpt-image] generating cover image (legacy): {fname}")
                cands = self.generate_candidates(
                    base_filename=fname, prompt=prompt, cover=True, n=1
                )
                self.finalize_candidate(candidate_path=cands[0], final_filename=fname)
                self.cleanup_candidates(base_filename=fname)

            else:
                self._placeholder(fname, f"[COVER {key}] {prompt}", cover=True)
