# BioEvidence Extraction Training v1

## Summary

This dataset supports query-focused structured evidence extraction from PubMed
titles and abstracts. It contains 60 query-PMID pairs across asthma, type 2
diabetes, statin prevention, melanoma immunotherapy, and dietary sodium topics.

The tracked source consists of a 20-row pilot plus a 40-row expansion. Labels
are model-assisted drafts and preserve `annotation_status=draft` explicitly.

## Schema and validation

Every row targets `ModelEvidenceExtraction` v1 and is checked for:

- valid JSON and Pydantic schema compliance;
- closed enum values and status-dependent nullability;
- exact verbatim outcome spans in the source abstract;
- unique annotation IDs and query-PMID pairs;
- deterministic PMID-level split ownership.

## Composition

| Evidence status | Rows |
| --- | ---: |
| direct | 8 |
| indirect | 29 |
| none | 23 |

| Split | Rows | Unique PMIDs | direct | indirect | none |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 46 | 43 | 4 | 25 | 17 |
| dev | 7 | 5 | 2 | 3 | 2 |
| test | 7 | 5 | 2 | 1 | 4 |

## Provenance

- Documents: PubMed titles and abstracts in the BioEvidence demo corpus.
- Pilot: model-assisted annotation with a second-model review.
- Expansion: produced in Claude Desktop, followed by repository schema,
  grounding, count, duplicate, and candidate-coverage validation.
- Split seed: 42.
- Split unit: PMID.

The source annotations, metadata, and aggregate split manifest are tracked under
`data/evaluations/evidence_extraction/`. Generated chat-format training files
remain under ignored `artifacts/` directories.

## Intended use and limitations

Accepted uses are pipeline validation, structured-output fine-tuning, and
held-out comparisons that name this dataset version. The dataset is not a
clinical benchmark or expert-gold biomedical corpus.

Known coverage is uneven: direct labels occur only in the diabetes, statin, and
sodium query groups. All labels remain drafts, and the seven-row test set makes
metric estimates unstable.

## Planned v2 expansion

The tracked `expansion_candidates.v2.jsonl` queue contains 60 currently
unlabeled query-PMID pairs after excluding all 60 v1 annotations. It samples 40
same-topic high-ranking and 20 same-topic broad documents, with no additional
cross-topic hard negatives. This targets the first adapter's conservative
direct/indirect failure mode; it does not prescribe the eventual label
distribution. The queue is not part of the v1 training dataset until a matching
validated annotation file is completed.
