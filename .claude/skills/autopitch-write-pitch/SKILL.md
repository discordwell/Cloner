---
name: autopitch-write-pitch
description: Generate a first-person self-addressing pitch script of a target duration. Use after autopitch-analyze-site has produced hypothesis.md.
---

# autopitch-write-pitch

Writes `pitch.txt` — the spoken script for the final video.

## When to use

After `hypothesis.md` exists in the run directory. The script reads it and
generates a time-budgeted pitch in the first-person self-addressing voice
("Hey [name] — yes, it's you...").

## How to invoke

```bash
python -m autopitch.scripts.write_pitch \
    --run autopitch/runs/{slug} \
    --name "Jane Doe" \
    --company "Acme Widgets" \
    [--role CEO] \
    [--duration 60]
```

## Tone + constraints

- First line must immediately establish "you pitching you" — make the viewer
  laugh at the premise before they can get defensive.
- Reference one specific item from the hypothesis so it feels hand-crafted.
- End with one clear ask ("reply and let's do 20 minutes about X").
- Plain spoken English — will be read by ElevenLabs TTS. No stage directions,
  no bracketed notes, no markdown, no emojis.
- Length is calibrated from `--duration` × ~2.4 words/sec (default ~150 words
  for 60s). Stay within ±10%.

## Where it fits in the pipeline

1. scrape_site.py → site.txt
2. autopitch-analyze-site → hypothesis.md
3. **autopitch-write-pitch** → pitch.txt   ← you are here
4. tts.py speak → pitch.mp3
