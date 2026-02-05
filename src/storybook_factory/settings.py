import os
from dataclasses import dataclass


def _bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.lower() in ("1", "true", "yes", "on")


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, default))


def _float(name: str, default: float) -> float:
    return float(os.getenv(name, default))


def _str(name: str, default: str) -> str:
    return os.getenv(name, default)


@dataclass(frozen=True)
class Settings:

    # --- Image generation ---
    image_model: str = _str("STORYBOOK_IMAGE_MODEL", "gpt-image-1")
    image_size: str = _str("STORYBOOK_IMAGE_SIZE", "1024x1536")
    image_quality: str = _str("STORYBOOK_IMAGE_QUALITY", "high")
    candidates_n: int = _int("STORYBOOK_CANDIDATES_N", 1)

    # --- Reviewer ---
    enable_reviewer: bool = _bool("STORYBOOK_ENABLE_REVIEWER", False)
    review_model: str = _str("STORYBOOK_REVIEW_MODEL", "gpt-4.1")
    review_downscale: int = _int("STORYBOOK_REVIEW_DOWNSCALE", 512)
    review_shuffle: bool = _bool("STORYBOOK_REVIEW_SHUFFLE", True)

    # --- Prompt behavior ---
    enforce_single_scene: bool = _bool("STORYBOOK_ENFORCE_SINGLE_SCENE", True)
    enforce_page_fill: bool = _bool("STORYBOOK_ENFORCE_PAGE_FILL", True)
    disable_prompt_opt: bool = _bool("STORYBOOK_DISABLE_PROMPT_OPT", False)

    # --- Output / debug ---
    keep_candidates: bool = _bool("STORYBOOK_KEEP_CANDIDATES", False)
    log_prompts: bool = _bool("STORYBOOK_LOG_PROMPTS", False)
    dry_run: bool = _bool("STORYBOOK_DRY_RUN", False)

    # --- Validation (cheap safety rails) ---
    validate_coverage: bool = _bool("STORYBOOK_VALIDATE_COVERAGE", True)
    min_coverage: float = _float("STORYBOOK_MIN_COVERAGE", 0.65)
    validate_grayscale: bool = _bool("STORYBOOK_VALIDATE_GRAYSCALE", True)
    validate_tiling: bool = _bool("STORYBOOK_VALIDATE_TILING", True)

    def summary(self) -> str:
        return (
            f"[storybook settings]\n"
            f"  image_model={self.image_model}\n"
            f"  image_size={self.image_size}\n"
            f"  image_quality={self.image_quality}\n"
            f"  candidates_n={self.candidates_n}\n"
            f"  reviewer={'on' if self.enable_reviewer else 'off'}\n"
            f"  keep_candidates={self.keep_candidates}\n"
            f"  dry_run={self.dry_run}"
        )


settings = Settings()
