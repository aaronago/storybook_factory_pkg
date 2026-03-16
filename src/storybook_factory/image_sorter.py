from __future__ import annotations

import json
import os
from pathlib import Path

from .image_provider import _google_genai, _google_types


class GeminiImageSorter:
    """
    Uses Gemini to classify and sort reference images into character buckets.
    """

    def __init__(self, google_client=None):
        """
        Args:
            google_client: An existing google.genai.Client. If None, one will be created.
        """
        self._google_client = google_client
        if self._google_client is None:
            if _google_genai is None:
                raise ImportError(
                    "google-genai is not installed. Run: pip install google-genai"
                )

            api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "GEMINI_API_KEY (or GOOGLE_API_KEY) env var is required."
                )

            self._google_client = _google_genai.Client(
                api_key=api_key,
                http_options={"api_version": "v1beta"},
                vertexai=False,
            )

    def get_reference_mapping(
        self,
        image_paths: list[Path],
        character_names: list[str],
        model: str = "gemini-3.1-flash-lite-preview",
    ) -> tuple[str, list[Path]]:
        """
        Groups the input image_paths by character and produces:
          1. A formatted REFERENCES string for the prompt.
          2. A re-ordered list of Path objects that matches those indices.

        Args:
            image_paths: List of local paths to reference images.
            character_names: Names of characters to look for.

        Returns:
            tuple: (ref_description_string, sorted_image_paths)
        """
        if not image_paths:
            return "", []

        # 1. Use the existing sorting logic to get indices
        results = self.sort_user_uploads(image_paths, character_names, model=model)

        # 2. Build the re-ordered list and the label string
        from collections import defaultdict

        sorted_paths: list[Path] = []
        char_to_ids = defaultdict(list)
        current_idx = 1

        # We iterate in the order of character_names for deterministic grouping
        for name in character_names:
            indices = results.get(name) or results.get(name.lower())
            if not indices:
                continue

            for idx in indices:
                if 0 <= idx < len(image_paths):
                    sorted_paths.append(image_paths[idx])
                    char_to_ids[name.capitalize()].append(f"[{current_idx}]")
                    current_idx += 1

        # Handle 'unknown' or any leftovers if they exist in the Gemini response
        if "unknown" in results:
            for idx in results["unknown"]:
                if 0 <= idx < len(image_paths):
                    sorted_paths.append(image_paths[idx])
                    char_to_ids["Style/Context"].append(f"[{current_idx}]")
                    current_idx += 1

        # 3. Format the final string
        ref_lines = []
        for char_name in sorted(char_to_ids.keys()):
            # Important: If it's a character from the list, we want it first usually,
            # but sorted keys is fine for the description block.
            ids_str = ", ".join(char_to_ids[char_name])
            ref_lines.append(f"{char_name}: {ids_str}")

        ref_description_string = "\n".join(ref_lines)
        return ref_description_string, sorted_paths

    def sort_user_uploads(
        self,
        image_paths: list[Path],
        character_names: list[str],
        model: str = "gemini-3.1-flash-lite-preview",
    ) -> dict[str, list[int]]:
        """
        Original sorting method (maintained for backward compatibility).
        Note: The prompt is updated to explicitly map to the names providing.
        """
        if not image_paths:
            return {name: [] for name in character_names}

        # Prepare the multimodal content list
        contents = []
        for path in image_paths:
            with open(path, "rb") as f:
                mime_type = "image/jpeg"
                if path.suffix.lower() == ".png":
                    mime_type = "image/png"

                contents.append(
                    _google_types.Part.from_bytes(data=f.read(), mime_type=mime_type)
                )

        # Instructions for the model
        sorting_prompt = (
            f"Identify the people or pets in these images. "
            f"Assign each image index (0-based) to one of these character names: {', '.join(character_names)}. "
            "Return ONLY a JSON object where keys are CHARACTER NAMES and values are lists of integers (indices). "
            "If an image is unclear or not one of characters, put it in 'unknown'."
        )
        contents.append(sorting_prompt)

        # Call the model with JSON response mode
        response = self._google_client.models.generate_content(
            model=model,
            contents=contents,
            config=_google_types.GenerateContentConfig(
                response_mime_type="application/json"
            ),
        )

        try:
            return json.loads(response.text)
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"Failed to parse Gemini sorting response: {response.text}"
            ) from e
