# Storybook Factory# storybook-factory



A command-line tool for generating illustrated storybooks with AI-powered image generation, character consistency, and PDF creation.A Python package to turn **YAML briefs + scene packs** into:



## Installation- `visual_bible.json` (style + cast)

- `page_prompts.json` (scene prompts)

```bash- `pipeline_config.json` (sizes, paths)

pip install -e .- A full **Lulu-ready book**: interior PDF, wrap-around cover PDF, and a ZIP package.

```

It is theme-agnostic and supports many different children, pets, and scenes.

## CLI CommandsYou feed it a *brief* and a *scene pack*; it does the rest.



### 1. `brief2json` - Convert Story Brief to JSON Config## Install



Convert a human-readable YAML story brief + theme pack into structured JSON configuration files (`page_prompts.json` and `visual_bible.json`).```bash

pip install .

**Usage:**```

```bash

storybook-factory brief2json \## Basic workflow

  --brief path/to/story.yaml \

  --theme theme_name \1. Create a YAML brief, e.g. `briefs/my_story.yaml`

  --out config_output_dir2. Use the built-in `dragon_realm` scene pack (or add your own under `scene_packs/`).

```3. Generate configs:



**Arguments:**```bash

- `--brief` (required): Path to the story brief YAML filestorybook-factory brief2json   --brief briefs/my_story.yaml   --theme dragon_realm   --out config

- `--theme` (required): Name of the theme/scene-pack YAML file (without extension)```

- `--out` (required): Output directory for generated JSON files

4. Build the book:

**Example:**

```bash```bash

storybook-factory brief2json \storybook-factory build   --config-dir config   --output-dir outputs   --image-provider mock

  --brief briefs/my_story.yaml \```

  --theme dragon_realm \

  --out out/configThen swap `mock` for `folder` when you have real PNGs (line-art pages + covers)

```in `assets/source_images/` with matching filenames from `page_prompts.json`.


---

### 2. `build` - Generate Complete Storybook

Generate character reference sheets, interior coloring pages, interior PDF, and a packaged ZIP file. This is the main command for creating a complete storybook.

**Usage:**
```bash
storybook-factory build \
  --config-dir config_directory \
  --output-dir output_directory \
  --image-provider gpt-image \
  --openai-interior-model model_name
```

**Arguments:**
- `--config-dir` (required): Directory containing `page_prompts.json` and `pipeline_config.json`
- `--output-dir` (required): Output directory for all generated artifacts
- `--assets-dir` (optional, default: `assets`): Folder with static assets (expects `assets/characters/<id>/...` for reference images)
- `--image-provider` (optional, default: `mock`): Image generation backend
  - `mock`: Generate placeholder images with text
  - `folder`: Copy existing images from assets
  - `gpt-image`: Generate images using OpenAI API
- `--openai-interior-model` (optional, default: `dall-e-2`): OpenAI model for interior pages and character sheets
- `--dry-run` (optional): Print prompts without actually generating images
- `--refs-source` (optional): Path to a previous output directory with `images/refs/*.png` to reuse character sheets

**Example:**
```bash
storybook-factory build \
  --config-dir out/config \
  --output-dir out/book_01 \
  --assets-dir assets \
  --image-provider gpt-image \
  --openai-interior-model gpt-image-1.5
```

**Example with refs from previous build:**
```bash
storybook-factory build \
  --config-dir out/config \
  --output-dir out/book_02 \
  --assets-dir assets \
  --image-provider gpt-image \
  --openai-interior-model gpt-image-1.5 \
  --refs-source out/book_01
```

**Output structure:**
```
out/book_01/
├── images/
│   ├── refs/          # Character reference sheets
│   ├── page_*.png     # Interior coloring pages
│   └── front_*.png    # Front-matter pages (if defined)
├── book/
│   └── interior.pdf   # Assembled interior PDF
└── book.zip           # Package containing all assets
```

---

### 3. `frontmatter` - Generate Front-Matter Pages Only

Generate only the front-matter pages (e.g., title page, introduction) without creating interior pages or covers.

**Usage:**
```bash
storybook-factory frontmatter \
  --config-dir config_directory \
  --output-dir output_directory \
  --image-provider gpt-image \
  --openai-interior-model model_name
```

**Arguments:**
- `--config-dir` (required): Directory containing `page_prompts.json`
- `--output-dir` (required): Output directory for front-matter images
- `--assets-dir` (optional, default: `assets`): Folder with static assets
- `--image-provider` (optional, default: `mock`): Image generation backend (`mock`, `folder`, `gpt-image`)
- `--openai-interior-model` (optional, default: `dall-e-2`): OpenAI model for front-matter pages

**Example:**
```bash
storybook-factory frontmatter \
  --config-dir out/config \
  --output-dir out/frontmatter_test \
  --assets-dir assets \
  --image-provider gpt-image \
  --openai-interior-model gpt-image-1.5
```

**Output structure:**
```
out/frontmatter_test/
└── images/
    └── front_*.png    # Front-matter pages
```

---

### 4. `refs` - Generate Character Reference Sheets Only

Generate only character reference sheets from the assets folder. These can be used to maintain character consistency across multiple storybooks.

**Usage:**
```bash
storybook-factory refs \
  --output-dir output_directory \
  --assets-dir assets_directory \
  --image-provider gpt-image \
  --openai-interior-model model_name
```

**Arguments:**
- `--output-dir` (required): Output directory for reference sheets
- `--assets-dir` (optional, default: `assets`): Assets folder (expects `assets/characters/<id>/...`)
- `--image-provider` (optional, default: `gpt-image`): Image generation backend (`mock`, `folder`, `gpt-image`)
- `--openai-interior-model` (optional, default: `gpt-image-1`): OpenAI model for character sheets
- `--config-dir` (optional): Path to directory containing `page_prompts.json` for character ID filtering

**Example:**
```bash
storybook-factory refs \
  --output-dir out/refs_test \
  --assets-dir assets \
  --image-provider gpt-image \
  --openai-interior-model gpt-image-1
```

**Example with character filtering:**
```bash
storybook-factory refs \
  --output-dir out/refs_test \
  --assets-dir assets \
  --config-dir out/config \
  --image-provider gpt-image \
  --openai-interior-model gpt-image-1
```

**Output structure:**
```
out/refs_test/
└── images/
    └── refs/
        ├── character_1.png
        ├── character_2.png
        └── ...
```

---

## Configuration Files

### `page_prompts.json`

Defines the content and prompts for all pages in the storybook.

**Structure:**
```json
{
  "front_matter": {
    "title_page": {
      "file": "front_title.png",
      "prompt": "A beautiful illustrated title page..."
    }
  },
  "interior_prompts": [
    {
      "page": 1,
      "file": "page_001.png",
      "title": "The Beginning",
      "prompt": "A descriptive prompt for the first page..."
    }
  ],
  "covers": {
    "front": {
      "file": "cover_front.png",
      "prompt": "Front cover illustration prompt..."
    },
    "back": {
      "file": "cover_back.png",
      "prompt": "Back cover illustration prompt..."
    }
  }
}
```

### `pipeline_config.json`

Defines pipeline settings like image dimensions and DPI.

**Structure:**
```json
{
  "interior_pixels": {
    "w": 1024,
    "h": 1536
  },
  "cover_pixels": {
    "w": 1536,
    "h": 1024
  },
  "dpi": 300,
  "trim_in": {
    "w": 5.5,
    "h": 8.5
  }
}
```

---

## Asset Structure

For character reference sheets, organize your assets as follows:

```
assets/
└── characters/
    ├── character_1_id/
    │   ├── reference/
    │   │   ├── photo_1.jpg
    │   │   ├── photo_2.jpg
    │   │   └── sketch.png
    │   └── kind.txt (optional: "child" or "pet")
    └── character_2_id/
        ├── reference/
        │   └── reference_image.png
        └── kind.txt
```

---

## Environment Variables

### Image Provider Settings

- `STORYBOOK_CANDIDATES_N`: Number of candidates to generate per page (default: `4`, max: `2`)
- `STORYBOOK_MAX_REGEN`: Maximum regeneration attempts (default: `2`)
- `STORYBOOK_KEEP_CANDIDATES`: Keep candidate images and review reports (default: `false`)
- `STORYBOOK_DISABLE_IMAGE_REVIEW`: Disable image review step (default: `false`)

### Image Generation Settings

- `STORYBOOK_INTERIOR_API_SIZE`: API size for interior images (default: `1024x1536`)
- `STORYBOOK_COVER_API_SIZE`: API size for cover images (default: `1536x1024`)
- `STORYBOOK_INTERIOR_QUALITY`: Quality for interior images (default: `high`)
- `STORYBOOK_COVER_QUALITY`: Quality for cover images (default: `high`)

### Review Settings

- `STORYBOOK_REVIEWER`: Reviewer mode (default: `off`)
  - `off`: Always accept first candidate
  - `basic`: Placeholder for future smart scoring

### OpenAI Settings

- `OPENAI_API_KEY`: Your OpenAI API key (required for `gpt-image` provider)

**Example:**
```bash
export OPENAI_API_KEY=sk-...
export STORYBOOK_CANDIDATES_N=2
export STORYBOOK_INTERIOR_API_SIZE=1024x1536
storybook-factory build --config-dir out/config --output-dir out/book_01 --image-provider gpt-image
```

---

## Quick Start

1. **Prepare your story brief:**
   ```bash
   # Create a story.yaml with your story outline
   ```

2. **Convert brief to JSON config:**
   ```bash
   storybook-factory brief2json \
     --brief briefs/my_story.yaml \
     --theme my_theme \
     --out out/config
   ```

3. **Prepare character assets (optional):**
   ```bash
   # Add reference images to assets/characters/<id>/reference/
   ```

4. **Generate the complete storybook:**
   ```bash
   storybook-factory build \
     --config-dir out/config \
     --output-dir out/my_book \
     --assets-dir assets \
     --image-provider gpt-image \
     --openai-interior-model gpt-image-1.5
   ```

5. **Find your output:**
   - Images: `out/my_book/images/`
   - PDF: `out/my_book/book/interior.pdf`
   - Package: `out/my_book/book.zip`

---

## Troubleshooting

### "Config directory not found"
- Ensure `--config-dir` contains `page_prompts.json` and `pipeline_config.json`

### "OpenAI API error"
- Check your `OPENAI_API_KEY` environment variable
- Ensure you have sufficient API quota

### "No character sheets found"
- Add character reference images to `assets/characters/<id>/reference/`
- Ensure images are in `.png`, `.jpg`, `.jpeg`, or `.webp` format

### Images not generating
- Try with `--image-provider mock` first to verify configuration
- Check pipeline configuration for valid model names
- Use `--dry-run` to validate prompts without generating

---

## License

[Your License Here]

## Support

For issues or questions, please open an issue on GitHub.
