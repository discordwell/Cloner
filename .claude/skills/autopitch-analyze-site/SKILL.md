---
name: autopitch-analyze-site
description: Produce a 1-3 bullet AI-opportunity hypothesis for a prospect's company from scraped homepage text. Use during the autopitch pipeline after scrape_site.py has written site.txt.
---

# autopitch-analyze-site

Generates `hypothesis.md` in a run directory from `site.txt`.

## When to use

Call this after `autopitch/scripts/scrape_site.py` has populated `{run_dir}/site.txt`.
It invokes the LLM with the template at `autopitch/prompts/analyze_site.txt` and
writes the structured hypothesis to `{run_dir}/hypothesis.md`.

## How to invoke

Run the underlying script directly — it handles prompt templating and LLM calls:

```bash
python -m autopitch.scripts.analyze_site \
    --run autopitch/runs/{slug} \
    --company "{Company Name}" \
    --url https://{company-url}
```

## Output format

A markdown file with:
- `**Business in one line:**` — single-sentence summary
- One to three `## Opportunity N: {title}` sections, each 2-3 sentences

## Quality bar

- Specific to their actual business model, not generic ("add a chatbot")
- Each opportunity implies a 3-6 month engagement, not a weekend hack
- If the site is thin, returns fewer opportunities framed as questions

## Where it fits in the pipeline

1. scrape_site.py → site.txt
2. **autopitch-analyze-site** → hypothesis.md   ← you are here
3. autopitch-write-pitch → pitch.txt
