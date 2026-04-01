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
        style_ref_dir: Path | None = None,
        character_roles: dict[str, str] | None = None,
        character_descs: dict[str, str] | None = None,
    ) -> tuple[str, list[Path]]:
        """
        Groups the input image_paths by character and produces:
          1. A formatted REFERENCES string for the prompt.
          2. A re-ordered list of Path objects that matches those indices.

        The returned path list always starts with the style reference image(s)
        so that [1] is always the master style reference as required by the prompt.

        Args:
            image_paths: List of local paths to reference images (characters only).
            character_names: Role labels to classify images into
                             (e.g. ["human_01", "human_02", "companion"]).
            style_ref_dir: Optional directory to load style reference images from.
            character_roles: Unused — kept for backward compatibility.
            character_descs: Optional mapping of role label → description string
                             (e.g. {"human_01": "4yo boy, curly hair", "companion": "small tan Chihuahua"}).
                             Passed to Gemini so it can match photos to roles by appearance.

        Returns:
            tuple: (ref_description_string, sorted_image_paths)
        """
        image_extensions = (".png", ".jpg", ".jpeg", ".webp")

        # 1. Collect style reference images.
        #    Resolution order:
        #      a) explicit style_ref_dir argument
        #      b) inferred from image_paths: characters live in <assets>/characters/,
        #         so <assets>/style_reference/ is two levels up from any character image
        #      c) fall back to detecting paths already containing 'assets/style_reference'
        resolved_style_dir: Path | None = None
        if style_ref_dir is not None:
            resolved_style_dir = Path(style_ref_dir)
        elif image_paths:
            # image_paths[0] is e.g. .../assets/characters/foo.jpg
            # parent       → .../assets/characters/
            # parent.parent → .../assets/
            candidate = Path(image_paths[0]).parent.parent / "style_reference"
            if candidate.exists():
                resolved_style_dir = candidate

        if resolved_style_dir is not None and resolved_style_dir.exists():
            style_ref_paths = sorted(
                p
                for p in resolved_style_dir.iterdir()
                if p.is_file() and p.suffix.lower() in image_extensions
            )
        else:
            style_ref_paths = [
                p for p in image_paths if "assets/style_reference" in str(p.absolute())
            ]

        # Character images = everything that isn't a style reference
        other_paths = [p for p in image_paths if p not in style_ref_paths]

        if not style_ref_paths and not other_paths:
            return "", []

        # 2. Use the existing sorting logic for character images only
        results = (
            self.sort_user_uploads(
                other_paths,
                character_names,
                model=model,
                character_descs=character_descs,
            )
            if other_paths
            else {}
        )

        # 3. Build the re-ordered list and the label string
        from collections import defaultdict

        sorted_paths: list[Path] = []
        char_to_ids = defaultdict(list)
        current_idx = 1

        # 3a. Add style references first (from assets/style_reference)
        for path in style_ref_paths:
            sorted_paths.append(path)
            char_to_ids["style_reference"].append(f"[{current_idx}]")
            current_idx += 1

        # Track which other_paths indices have already been assigned so each photo
        # appears under exactly one role, even if Gemini duplicates indices across roles.
        used_indices: set[int] = set()

        # 3b. Add 'unknown' from the Gemini sorting to the same bucket
        if "unknown" in results:
            for idx in results["unknown"]:
                if 0 <= idx < len(other_paths) and idx not in used_indices:
                    used_indices.add(idx)
                    sorted_paths.append(other_paths[idx])
                    char_to_ids["style_reference"].append(f"[{current_idx}]")
                    current_idx += 1

        # 3c. We iterate in the order of character_names for deterministic grouping
        for name in character_names:
            indices = results.get(name) or results.get(name.lower())
            if not indices:
                continue

            # Role labels ARE the labels — use them directly (no remapping needed)
            label = name

            for idx in indices:
                if 0 <= idx < len(other_paths) and idx not in used_indices:
                    used_indices.add(idx)
                    sorted_paths.append(other_paths[idx])
                    char_to_ids[label].append(f"[{current_idx}]")
                    current_idx += 1

        # 4. Format the final string
        # Build the per-character index lines
        char_lines: list[str] = []

        # Ensure style_reference is always first in the text output if it exists
        if "style_reference" in char_to_ids:
            ids_str = ", ".join(char_to_ids["style_reference"])
            char_lines.append(f"style_reference: {ids_str}")

        # Preserve insertion order for role labels (not alpha-sorted)
        for label in char_to_ids:
            if label == "style_reference":
                continue
            ids_str = ", ".join(char_to_ids[label])
            char_lines.append(f"{label}: {ids_str}")

        # Compose final string
        ref_description_string = "REFERENCES: " + "\n".join(char_lines)
        return ref_description_string, sorted_paths

    def sort_user_uploads(
        self,
        image_paths: list[Path],
        character_names: list[str],
        model: str = "gemini-3.1-flash-lite-preview",
        character_descs: dict[str, str] | None = None,
    ) -> dict[str, list[int]]:
        """
        Uses Gemini vision to classify each image into a role bucket.
        character_descs maps role label → appearance description so Gemini
        can match photos to roles (e.g. {"human_01": "4yo boy, curly hair"}).
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

        # Build role descriptions for the prompt
        if character_descs:
            role_lines = "\n".join(
                f"  {role}: {character_descs[role]}"
                for role in character_names
                if role in character_descs
            )
            sorting_prompt = (
                f"You are sorting photos into character roles for a children's storybook.\n"
                f"Roles and their descriptions:\n{role_lines}\n\n"
                f"Assign each image index (0-based) to the role whose description best matches "
                f"the person or pet shown. Roles: {', '.join(character_names)}.\n"
                "Return ONLY a JSON object where keys are role labels and values are lists of "
                "integer indices. If an image doesn't match any role, put it in 'unknown'."
            )
        else:
            sorting_prompt = (
                f"Identify the people or pets in these images. "
                f"Assign each image index (0-based) to one of these roles: {', '.join(character_names)}. "
                "Return ONLY a JSON object where keys are role labels and values are lists of integers (indices). "
                "If an image is unclear or not one of the roles, put it in 'unknown'."
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
