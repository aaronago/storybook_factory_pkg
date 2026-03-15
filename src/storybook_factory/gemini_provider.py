from __future__ import annotations

import os
from pathlib import Path
from PIL import Image

try:
    from google import genai as _google_genai
    from google.genai import types as _google_types
except ImportError:
    _google_genai = None  # type: ignore[assignment]
    _google_types = None  # type: ignore[assignment]

class GeminiProvider:
    \"\"\"
    Dedicated provider for Google Gemini/Imagen image generation.
    \"\"\"

    def __init__(self, api_key: str | None = None):
        if _google_genai is None or _google_types is None:
            raise ImportError(
                "google-genai is not installed. Run: pip install google-genai"
            )

        self.api_key = api_key or os.getenv(\"GEMINI_API_KEY\") or os.getenv(\"GOOGLE_API_KEY\")
        if not self.api_key:
            raise RuntimeError(
                \"GEMINI_API_KEY (or GOOGLE_API_KEY) env var is required for Gemini models.\"
            )

        self.client = _google_genai.Client(
            api_key=self.api_key,
            http_options={\"api_version\": \"v1beta\"},
            vertexai=False,
        )

    def generate_candidates(
        self,
        *,
        model_name: str,
        prompt: str,
        n: int,
        aspect_ratio: str = \"3:4\",
        refs: list[Path] | None = None,
    ) -> list[Image.Image]:
        \"\"\"
        Generate candidate PIL Images via Google Gemini (multimodal) or Imagen.
        \"\"\"
        clean_model = model_name.removeprefix(\"models/\")
        refs = refs or []

        try:
            # If reference images are provided, use generate_content (multimodal)
            if refs:
                contents = []
                for ref_path in refs:
                    with open(ref_path, \"rb\") as f:
                        image_bytes = f.read()
                    
                    mime_type = \"image/jpeg\"
                    if ref_path.suffix.lower() == \".png\":
                        mime_type = \"image/png\"

                    contents.append(
                        _google_types.Part.from_bytes(
                            data=image_bytes, mime_type=mime_type
                        )
                    )

                full_prompt = (
                    \"Generate a new image in the same style, character design, and color palette as \"
                    \"the provided reference images. \" + prompt
                )
                contents.append(full_prompt)

                response = self.client.models.generate_content(
                    model=clean_model,
                    contents=contents,
                    config=_google_types.GenerateContentConfig(
                        response_modalities=[\"IMAGE\"],
                        image_config=_google_types.ImageConfig(aspect_ratio=aspect_ratio),
                        candidate_count=n,
                    ),
                )

                if not response.candidates:
                    raise RuntimeError(\"No candidates returned from Gemini multimodal generation.\")

                pil_images = []
                for candidate in response.candidates:
                    images_in_part = [
                        p.as_image()
                        for p in (candidate.content.parts or [])
                        if p.inline_data
                    ]
                    if images_in_part:
                        pil_images.append(images_in_part[0])

                if not pil_images:
                    raise RuntimeError(\"No images found in Gemini multimodal response.\")
                return pil_images

            else:
                # Standard Imagen path for text-to-image
                response = self.client.models.generate_images(
                    model=clean_model,
                    prompt=prompt,
                    config=_google_types.GenerateImagesConfig(
                        number_of_images=n,
                        aspect_ratio=aspect_ratio,
                        safety_filter_level=\"BLOCK_LOW_AND_ABOVE\",
                    ),
                )

                if not response.generated_images:
                    raise RuntimeError(\"No images generated. Check safety filters or prompt.\")

                return [gen_image.image for gen_image in response.generated_images]

        except Exception as e:
            raise RuntimeError(f\"Gemini image generation failed: {e}\") from e
