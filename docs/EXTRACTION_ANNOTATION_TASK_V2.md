# Evidence extraction annotation task v2

## Goal

Create 60 model-assisted draft annotations for the second evidence-extraction
expansion. Follow `docs/EVIDENCE_ANNOTATION_GUIDE.md` and the v1 Pydantic schema
exactly. Do not force any target class: label each query-PMID pair from its title
and abstract, even though this batch was selected to improve direct/indirect
coverage.

## Inputs

- Candidates: `data/evaluations/evidence_extraction/expansion_candidates.v2.jsonl`
- Documents: `data/corpora/demo/processed/demo.documents.jsonl`
- Schema implementation: `src/bioevidence/schemas/model_evidence.py`
- Annotation guide: `docs/EVIDENCE_ANNOTATION_GUIDE.md`
- Format example: `data/evaluations/evidence_extraction/expansion_annotations.v1.jsonl`

## Required output

Write exactly one JSON object per candidate to:

`data/evaluations/evidence_extraction/expansion_annotations.v2.jsonl`

Each row must contain:

```json
{
  "id": "candidate id unchanged",
  "query": "candidate query unchanged",
  "pmid": "candidate PMID unchanged",
  "annotation_status": "draft",
  "extraction": {}
}
```

Populate `extraction` according to the v1 schema. In particular:

- `none` requires null population, intervention, comparator, and summary, plus
  an empty outcomes array;
- `direct` requires at least one outcome;
- every `evidence_span` must be copied verbatim and contiguously from the
  corresponding abstract;
- keep status query-focused: a rigorous study can still be indirect or none for
  the supplied query.

## Validation

Run this command after writing the file:

```powershell
python scripts/validate_extraction_annotations.py `
  --dataset data/evaluations/evidence_extraction/expansion_annotations.v2.jsonl `
  --candidates data/evaluations/evidence_extraction/expansion_candidates.v2.jsonl `
  --data-dir data/corpora/demo
```

The command must report 60 validated annotations and `Candidate coverage:
60/60`. Fix all JSON, schema, duplicate, PMID, grounding, and coverage errors
before declaring the task complete.
