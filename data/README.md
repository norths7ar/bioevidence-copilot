# Data Layout

This repository keeps small, reviewable demo and evaluation artifacts under `data/` and
ignores large local downloads, caches, and raw benchmark archives.

Tracked:

- `data/corpora/`: reproducible local document corpora
- `data/evaluations/`: evaluation JSONL files and small example reports

Ignored:

- `data/cache/`
- downloaded BioASQ zip files and expanded raw benchmark folders under `tmp/`
- local raw PubMed XML/JSON fetch artifacts unless intentionally moved into a
  tracked corpus folder
