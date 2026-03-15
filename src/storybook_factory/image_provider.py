# image_provider.py
from __future__ import annotations

import base64
import os
import shutil
from collections.abc import Iterable
from pathlib import Path

import numpy as np
from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont

try:
    from google import genai as _google_genai
    from google.genai import types as _google_types
except ImportError:
    _google_genai = None  # type: ignore[assignment]
    _google_types = None  # type: ignore[assignment]

DEFAULT_FONT = ImageFont.load_default()


def _is_google(model: str | None) -> bool:
    """Return True if the model name looks like a Google image model (Gemini or Imagen)."""
    if not model:
        return False
    lower = model.lower()
    return "gemini" in lower or "imagen" in lower


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


class ImageProvider:
    """
    Handles creation of page/cover PNGs for the pipeline.

    mode:
      - "mock"      placeholder images with prompt text
      - "folder"    copies existing PNGs from assets_dir
      - "gpt-image" generates via OpenAI Images API
                    (model names that do NOT contain "gemini")
      - "gpt-image" with a gemini interior/cover model → routes to Google Imagen API
    """

    def __init__(
        self,
        out_dir: Path,
        interior_px,
        cover_px,
        mode: str = "mock",
        cover_model: str | None = None,
        interior_model: str | None = None,
        assets_dir: Path | None = None,
        dry_run: bool = False,
        image_quality: str = "low",
    ):
        self.out_dir = out_dir
        self.interior_px = self._normalize_px(interior_px, name="interior_px")
        self.cover_px = self._normalize_px(cover_px, name="cover_px")
        self.mode = mode
        self.cover_model = cover_model
        self.interior_model = interior_model
        self.assets_dir = assets_dir
        self.dry_run = dry_run

        # Allow overrides via env
        self.interior_api_size = os.getenv("STORYBOOK_INTERIOR_API_SIZE", "1024x1536")
        self.cover_api_size = os.getenv("STORYBOOK_COVER_API_SIZE", "1024x1536")
        self.interior_quality = os.getenv("STORYBOOK_INTERIOR_QUALITY", image_quality)
        self.cover_quality = os.getenv("STORYBOOK_COVER_QUALITY", image_quality)

        # Prevent "hang forever"
        # You can override:
        #   STORYBOOK_OPENAI_TIMEOUT=90
        #   STORYBOOK_OPENAI_RETRIES=1
        if self.mode == "gpt-image":
            timeout_s = float(os.getenv("STORYBOOK_OPENAI_TIMEOUT", "120"))
            max_retries = int(os.getenv("STORYBOOK_OPENAI_RETRIES", "2"))
            self.client = OpenAI(timeout=timeout_s, max_retries=max_retries)
        else:
            self.client = None

        # Google Gemini client — initialised lazily if either model is Gemini
        self._google_client = None
        if _is_google(self.interior_model) or _is_google(self.cover_model):
            if _google_genai is None:
                raise ImportError(
                    "google-genai is not installed. Run: pip install google-genai"
                )
            gemini_api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            if not gemini_api_key:
                raise RuntimeError(
                    "GEMINI_API_KEY (or GOOGLE_API_KEY) env var is required for Gemini models."
                )
            self._google_client = _google_genai.Client(
                api_key=gemini_api_key,
                http_options={"api_version": "v1beta"},
                vertexai=False,
            )

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
        out_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(out_path, "PNG")
        return out_path

    def _copy_from_assets(self, filename: str) -> Path | None:
        if not self.assets_dir:
            return None
        src = self.assets_dir / filename
        if src.exists():
            dst = self.out_dir / filename
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            return dst
        return None

    # ----------------------------
    # Candidate generation
    # ----------------------------

    def generate_candidates(
        self,
        *,
        base_filename: str,
        prompt: str,
        cover: bool = False,
        n: int = 1,
        reference_images: list[Path] | None = None,
    ) -> list[Path]:
        """
        Generate up to 2 candidate images for a given prompt and return their paths.
        Routes to Google Imagen API when the model name contains "gemini",
        otherwise uses the OpenAI Images API.
        """
        # HARD CAP: never exceed 2 candidates
        n = max(1, min(int(n), 2))

        if self.mode != "gpt-image":
            # For mock/folder mode, just create one placeholder candidate.
            return [self._placeholder(base_filename, prompt, cover=cover)]

        if cover:
            api_size = self.cover_api_size
            target_w, target_h = self.cover_px
            model_name = self.cover_model
            quality = "high"
        else:
            api_size = self.interior_api_size
            target_w, target_h = self.interior_px
            model_name = self.interior_model
            quality = self.interior_quality

        if not model_name:
            raise RuntimeError("No model configured for this image type.")

        refs = [Path(p) for p in (reference_images or []) if Path(p).exists()]

        if self.dry_run:
            print(
                f"[dry-run] would generate {n} candidates for {base_filename} "
                f"model={model_name} size={api_size} refs={len(refs)}"
            )
            print(prompt)
            print("-" * 80)
            return [
                self._placeholder(base_filename, f"[DRY RUN] {prompt}", cover=cover)
            ]

        if _is_google(model_name):
            return self._generate_candidates_gemini(
                base_filename=base_filename,
                prompt=prompt,
                model_name=model_name,
                n=n,
                refs=refs,
                target_w=target_w,
                target_h=target_h,
                cover=cover,
            )
        else:
            return self._generate_candidates_openai(
                base_filename=base_filename,
                prompt=prompt,
                model_name=model_name,
                api_size=api_size,
                quality=quality,
                n=n,
                refs=refs,
                target_w=target_w,
                target_h=target_h,
                cover=cover,
            )

    def _generate_candidates_gemini(
        self,
        *,
        base_filename: str,
        prompt: str,
        model_name: str,
        n: int,
        refs: list[Path],
        target_w: int,
        target_h: int,
        cover: bool,
    ) -> list[Path]:
        """
        Generate candidates via Google Gemini (multimodal) or Imagen using the SDK.
        """
        if _google_genai is None or _google_types is None:
            raise ImportError(
                "google-genai is not installed. Run: pip install google-genai"
            )
        if self._google_client is None:
            raise RuntimeError("Google client not initialised.")

        # Use model name from args directly
        clean_model = model_name.removeprefix("models/")

        try:
            # If reference images are provided, use generate_content (multimodal)
            if refs:
                contents = []
                for ref_path in refs:
                    with open(ref_path, "rb") as f:
                        image_bytes = f.read()
                    # Determine mime type from extension
                    mime_type = "image/jpeg"
                    if ref_path.suffix.lower() == ".png":
                        mime_type = "image/png"

                    contents.append(
                        _google_types.Part.from_bytes(
                            data=image_bytes, mime_type=mime_type
                        )
                    )

                # Prepend a style instruction if refs are provided
                full_prompt = (
                    "Generate a new image in the same style, character design, and color palette as "
                    "the provided reference images. " + prompt
                )
                contents.append(full_prompt)

                response = self._google_client.models.generate_content(
                    model=clean_model,
                    contents=contents,
                    config=_google_types.GenerateContentConfig(
                        response_modalities=["IMAGE"],
                        image_config=_google_types.ImageConfig(aspect_ratio="3:4"),
                        candidate_count=n,
                    ),
                )

                # Process multimodal output
                out_paths: list[Path] = []
                # generate_content returns a list of candidates in response.candidates
                if not response.candidates:
                    raise RuntimeError(
                        "No candidates returned from Gemini multimodal generation."
                    )

                # Mapping candidates to files
                for i, candidate in enumerate(response.candidates):
                    # Look for image in candidate parts
                    images_in_part = [
                        p.as_image()
                        for p in (candidate.content.parts or [])
                        if p.inline_data
                    ]
                    if not images_in_part:
                        continue

                    # Take the first image found in the parts
                    gen_image = images_in_part[0]
                    cand_name = self._candidate_name(base_filename, i)
                    out_path = self.out_dir / cand_name
                    out_path.parent.mkdir(parents=True, exist_ok=True)

                    # The SDK Image object needs to be saved first, then opened with PIL
                    gen_image.save(str(out_path))

                    # Standard PIL resize/convert for the book pipeline
                    res_img = (
                        Image.open(out_path)
                        .convert("RGB")
                        .resize((target_w, target_h), Image.LANCZOS)
                    )
                    res_img.save(out_path, "PNG")
                    out_paths.append(out_path)

                if not out_paths:
                    raise RuntimeError(
                        "No images found in Gemini multimodal response parts."
                    )

                return out_paths

            else:
                # Text-to-image: use generate_content for image preview models
                response = self._google_client.models.generate_content(
                    model=clean_model,
                    contents=prompt,
                    config=_google_types.GenerateContentConfig(
                        response_modalities=["IMAGE"],
                        image_config=_google_types.ImageConfig(aspect_ratio="3:4"),
                        candidate_count=n,
                    ),
                )

                # Process candidates output
                out_paths: list[Path] = []
                if not response.candidates:
                    raise RuntimeError("No candidates returned from Gemini generation.")

                for i, candidate in enumerate(response.candidates):
                    # Look for image in candidate parts
                    images_in_part = [
                        p.as_image()
                        for p in (candidate.content.parts or [])
                        if p.inline_data
                    ]
                    if not images_in_part:
                        continue

                    gen_image = images_in_part[0]
                    cand_name = self._candidate_name(base_filename, i)
                    out_path = self.out_dir / cand_name
                    out_path.parent.mkdir(parents=True, exist_ok=True)

                    # Save SDK image then process with PIL
                    gen_image.save(str(out_path))

                    # Standard PIL resize/convert for the book pipeline
                    img = (
                        Image.open(out_path)
                        .convert("RGB")
                        .resize((target_w, target_h), Image.LANCZOS)
                    )
                    img.save(out_path, "PNG")
                    out_paths.append(out_path)

                if not out_paths:
                    raise RuntimeError("No images found in Gemini response parts.")

                return out_paths

        except Exception as e:
            raise RuntimeError(f"Gemini image generation failed: {e}") from e

    def _generate_candidates_openai(
        self,
        *,
        base_filename: str,
        prompt: str,
        model_name: str,
        api_size: str,
        quality: str,
        n: int,
        refs: list[Path],
        target_w: int,
        target_h: int,
        cover: bool,
    ) -> list[Path]:
        """Generate candidates via OpenAI Images API."""
        if self.client is None:
            raise RuntimeError(
                "OpenAI client not initialized; mode must be 'gpt-image'."
            )

        # --- With reference images: use edit endpoint with a blank canvas + refs ---
        if refs:
            import io

            # Create a blank canvas matching the API size
            w_str, h_str = api_size.lower().split("x", 1)
            w, h = int(w_str), int(h_str)

            noise = np.random.randint(250, 255, (h, w, 3), dtype=np.uint8)
            blank = Image.fromarray(noise)
            buf = io.BytesIO()
            blank.save(buf, format="PNG")
            buf.seek(0)

            try:
                buf.name = "blank.png"  # type: ignore[attr-defined]
            except Exception:
                pass

            files = [buf]
            opened_files = []

            try:
                for rp in refs:
                    f = open(rp, "rb")
                    opened_files.append(f)
                    files.append(f)

                result = self.client.images.edit(
                    model=model_name,
                    image=files,  # blank first, then refs
                    prompt=prompt,
                    size=api_size,
                    n=n,
                    quality=quality,
                    output_format="png",
                )
            finally:
                for f in opened_files:
                    try:
                        f.close()
                    except Exception:
                        pass
                try:
                    buf.close()
                except Exception:
                    pass

        # --- No reference images: standard generation ---
        else:
            result = self.client.images.generate(
                model=model_name,
                prompt=prompt,
                size=api_size,
                n=n,
                quality=quality,
                output_format="png",
            )

        # Write + resize outputs
        out_paths: list[Path] = []
        for i, data in enumerate(result.data):
            b64_data = data.b64_json
            img_bytes = base64.b64decode(b64_data)

            cand_name = self._candidate_name(base_filename, i)
            out_path = self.out_dir / cand_name
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(img_bytes)

            img = (
                Image.open(out_path)
                .convert("RGB")
                .resize((target_w, target_h), Image.LANCZOS)
            )
            img.save(out_path, "PNG")

            out_paths.append(out_path)

        return out_paths

    # ----------------------------
    # Edits (UPGRADED: multi-image + out_subdir)
    # ----------------------------

    def apply_edit(
        self,
        *,
        input_images: list[Path],
        prompt: str,
        out_filename: str,
        cover: bool = False,
        out_subdir: str | None = None,
    ) -> Path:
        """
        Optional edit pass.
        UPGRADED:
          - accepts multiple input images (identity is much better for humans)
          - supports writing to an output subdirectory (e.g. refs/)
          - routes to Google Imagen API when the model name contains "gemini"
        """
        if self.mode != "gpt-image":
            # In non-gpt mode, just return a placeholder.
            return self._placeholder(
                out_filename, f"[EDIT SKIPPED] {prompt}", cover=cover
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
            raise RuntimeError("No model configured for this image type.")

        if self.dry_run:
            print(
                f"[dry-run] would edit {len(input_images)} images into {out_filename} "
                f"using model={model_name} size={api_size}"
            )
            print(prompt)
            print("-" * 80)
            return self._placeholder(
                out_filename, f"[DRY RUN EDIT] {prompt}", cover=cover
            )

        base_dir = self.out_dir if not out_subdir else (self.out_dir / out_subdir)
        base_dir.mkdir(parents=True, exist_ok=True)
        out_path = base_dir / out_filename

        if _is_google(model_name):
            # Google Gemini/Imagen edit: pass input images as reference_images via generate_content
            if _google_genai is None or _google_types is None:
                raise ImportError("google-genai is not installed.")
            if self._google_client is None:
                raise RuntimeError("Google client not initialised.")

            clean_model = model_name.removeprefix("models/")

            try:
                # If input images are provided, use generate_content (multimodal)
                if input_images:
                    contents = []
                    for input_path in input_images:
                        with open(input_path, "rb") as f:
                            image_bytes = f.read()

                        mime_type = "image/jpeg"
                        if input_path.suffix.lower() == ".png":
                            mime_type = "image/png"

                        contents.append(
                            _google_types.Part.from_bytes(
                                data=image_bytes, mime_type=mime_type
                            )
                        )

                    # Prepend a style/edit instruction
                    full_prompt = (
                        "Generate a new image in the same style, character design, and color palette as "
                        "the provided reference images. " + prompt
                    )
                    contents.append(full_prompt)

                    response = self._google_client.models.generate_content(
                        model=clean_model,
                        contents=contents,
                        config=_google_types.GenerateContentConfig(
                            response_modalities=["IMAGE"],
                            image_config=_google_types.ImageConfig(aspect_ratio="3:4"),
                            candidate_count=1,
                        ),
                    )

                    if not response.candidates:
                        raise RuntimeError(
                            "No candidates returned from Gemini multimodal generation."
                        )

                    # Look for image in candidate parts
                    images_in_part = [
                        p.as_image()
                        for p in (response.candidates[0].content.parts or [])
                        if p.inline_data
                    ]
                    if not images_in_part:
                        raise RuntimeError(
                            "No images found in Gemini multimodal response."
                        )

                    # Save the SDK image object, then open with PIL
                    images_in_part[0].save(str(out_path))

                else:
                    # Standard Imagen path for text-to-image
                    response = self._google_client.models.generate_images(
                        model=clean_model,
                        prompt=prompt,
                        config=_google_types.GenerateImagesConfig(
                            number_of_images=1,
                            aspect_ratio="3:4",
                            safety_filter_level="BLOCK_LOW_AND_ABOVE",
                        ),
                    )
                    if not response.generated_images:
                        raise RuntimeError("No images generated.")

                    response.generated_images[0].image.save(str(out_path))

                # Standard PIL resize/convert for the book pipeline
                img = (
                    Image.open(out_path)
                    .convert("RGB")
                    .resize((target_w, target_h), Image.LANCZOS)
                )
                img.save(out_path, "PNG")

            except Exception as e:
                raise RuntimeError(f"Gemini image generation failed: {e}") from e

        else:
            # OpenAI edit
            if self.client is None:
                raise RuntimeError(
                    "OpenAI client not initialized; mode must be 'gpt-image'."
                )
            files = []
            try:
                for p in input_images:
                    files.append(open(p, "rb"))

                result = self.client.images.edit(
                    model=model_name,
                    image=files,
                    prompt=prompt,
                    size=api_size,
                )
            finally:
                for f in files:
                    try:
                        f.close()
                    except Exception:
                        pass

            b64_data = result.data[0].b64_json
            img_bytes = base64.b64decode(b64_data)
            out_path.write_bytes(img_bytes)

            img = (
                Image.open(out_path)
                .convert("RGB")
                .resize((target_w, target_h), Image.LANCZOS)
            )
            img.save(out_path, "PNG")
        return out_path

    def finalize_candidate(
        self, candidate_path: Path, final_filename: str, *, cover: bool
    ) -> None:
        img = Image.open(candidate_path)
        if cover:
            img = img.convert("RGB")
            img.save(self.out_dir / final_filename, "PNG")
        else:
            img = img.convert("L")
            img.save(self.out_dir / final_filename)

    def cleanup_candidates(self, *, base_filename: str) -> None:
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
    # Legacy API (kept)
    # ----------------------------

    def render_interior(self, interior_prompts: Iterable[dict]) -> None:
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
        # Covers disabled in your current direction, but keeping legacy method
        # doesn't hurt. You can delete later.
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
