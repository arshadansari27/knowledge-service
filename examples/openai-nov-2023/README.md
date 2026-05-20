# Demo corpus — OpenAI board weekend, November 2023

A small bundled corpus for `scripts/demo.py`. Eight short documents covering the firing and rapid reinstatement of Sam Altman as CEO of OpenAI between Friday 17 November and Wednesday 22 November 2023.

The events are deliberately chosen because they exercise the parts of the system that are hard to demonstrate on flat news: same-subject same-predicate value-deltas across a five-day window, multi-source agreement on stable facts, and rich entity overlap (Sam Altman, Greg Brockman, Mira Murati, Emmett Shear, Microsoft, the OpenAI board).

## What's in the corpus

| File | Date | Source type | What it asserts |
|------|------|-------------|-----------------|
| `01-board-statement.md` | 2023-11-17 | press release | Board fires Altman; Murati is interim CEO; Brockman steps down as chairman, retained as president |
| `02-altman-tweet.md` | 2023-11-17 | social post | Altman's terse public response |
| `03-news-coverage.md` | 2023-11-17 | article | Third-party reporting consolidating the firing facts |
| `04-brockman-resigns.md` | 2023-11-17 | social post | Brockman announces he is leaving OpenAI |
| `05-employee-letter.md` | 2023-11-20 | open letter | Letter from 700+ employees demanding board resignation |
| `06-microsoft-hire.md` | 2023-11-20 | press release | Microsoft announces Altman and Brockman will lead a new AI group |
| `07-shear-interim.md` | 2023-11-20 | social post | Emmett Shear announced as new interim CEO, replacing Murati |
| `08-altman-returns.md` | 2023-11-22 | press release | Altman reinstated as CEO; new board composition |

## What the demo exercises

- **Multi-source consolidation (Noisy-OR).** Several documents agree that Sam Altman *is* the CEO of OpenAI — at the start of the window, and again at the end. Their confidences combine.
- **Same-predicate value-delta contradictions** are visible in the extracted graph: across the corpus, OpenAI's CEO predicate resolves to Sam Altman → Mira Murati → Emmett Shear → Sam Altman without `valid_until` bounds. The current contradiction detector (same predicate, different objects) should fire on these pairs. Numerical contradictions (e.g. revaluations, guidance) are *not* exercised here — that's deliberately left for the Phase 2 wedge.
- **Entity overlap and coreference.** "Altman", "Sam Altman", and "Samuel H. Altman" appear across documents and should converge on a single canonical entity via the NLP pre-pass and coreference phase.
- **Trust tiers.** Mixing primary-source press releases (high confidence), social posts (lower), and third-party reporting (mid) lets the named-graph trust labels (`extracted`, `asserted`, etc.) be inspected.

## Licensing and provenance

Every document in this corpus is **paraphrased synthesis of publicly reported events** written specifically as demo content. No paragraph is copied verbatim from any news source or press release. Each file's frontmatter records the kind of public statement or coverage it is *based on* — those original statements are attributable and were public on the dates indicated. The paraphrased text in these files is released under the same MIT license as the rest of this repository.

If you want to swap the corpus for the live Wikipedia article on the same events, see `examples/README.md` (TODO — not in Phase 1) for the recommended ingestion command.

## How the documents are structured

Each document has a YAML frontmatter block (parsed by `scripts/demo.py`) followed by markdown prose:

```yaml
---
title: ...
url: demo://openai-nov-2023/<slug>
source_type: press_release | article | social_post | open_letter
publication: <attributed source>
published_at: YYYY-MM-DD
tags: [...]
---
```

The demo script reads the file, extracts the frontmatter as `/api/content` request fields, sends the markdown body as `raw_text` with empty `knowledge`, and lets the LLM extraction populate the graph.
