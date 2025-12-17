# storybook-factory

A Python package to turn **YAML briefs + scene packs** into:

- `visual_bible.json` (style + cast)
- `page_prompts.json` (scene prompts)
- `pipeline_config.json` (sizes, paths)
- A full **Lulu-ready book**: interior PDF, wrap-around cover PDF, and a ZIP package.

It is theme-agnostic and supports many different children, pets, and scenes.
You feed it a *brief* and a *scene pack*; it does the rest.

## Install

```bash
pip install .
```

## Basic workflow

1. Create a YAML brief, e.g. `briefs/my_story.yaml`
2. Use the built-in `dragon_realm` scene pack (or add your own under `scene_packs/`).
3. Generate configs:

```bash
storybook-factory brief2json   --brief briefs/my_story.yaml   --theme dragon_realm   --out config
```

4. Build the book:

```bash
storybook-factory build   --config-dir config   --output-dir outputs   --image-provider mock
```

Then swap `mock` for `folder` when you have real PNGs (line-art pages + covers)
in `assets/source_images/` with matching filenames from `page_prompts.json`.
