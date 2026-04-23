---
name: autopitch-cartoonify-prompt
description: Build the exact ChatGPT prompt for the Pixar portrait and scene-cartoonify steps. Use when the autopitch agent is driving chatgpt.com via MCP chrome tools.
---

# autopitch-cartoonify-prompt

Returns the prompt strings to paste into ChatGPT for the two cartoonify turns.

## When to use

During the cartoonify stages of the autopitch pipeline — the autopitch agent
reads a template from `autopitch/prompts/` and fills in company-specific hints
before sending to ChatGPT.

## How to use

Read the two prompt files and substitute `{company}` and `{business_hint}`:

- `autopitch/prompts/cartoonify_portrait.txt` — for the first turn (raw photo → Pixar portrait)
- `autopitch/prompts/cartoonify_scene.txt` — for the second turn (portrait + logo → scene)

The `{business_hint}` substitution in the scene prompt should be a short
description of the likely environment (e.g. "a sunlit craft-bakery front with
a chalkboard menu" or "a modern SaaS office with a glass reception desk").
Derive this from the hypothesis.md or the scraped site text.

## Guardrails baked into the prompt

- "Generate ONLY this one image" — blocks ChatGPT from producing grids.
- Preservation clauses for identity, ethnicity, age.
- Face-visibility and mouth-unobstructed requirements for the scene (so Atlas
  lip-sync has a clean target in stage 11).

## CRITICAL: Avoid ChatGPT Pro mode

Before submitting, confirm the model selector shows the standard GPT-4o / GPT-5
mode, NOT "GPT-5 Pro" or "extended thinking." Pro mode will rewrite the prompt
aggressively, take 60+s, and often produce artifacts that violate the "only
this one image" guardrail.

## Where it fits in the pipeline

1. find_photo.py → photo_raw.jpg
2. scrape_site.py → logo.png
3. **Cartoonify portrait** (ChatGPT via MCP) ← you are here
4. **Cartoonify scene** (ChatGPT via MCP, attaches logo) ← and here
5. lipsync.py → final.mp4
