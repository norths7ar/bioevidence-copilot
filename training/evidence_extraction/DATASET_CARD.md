# BioEvidence Extraction Training Datasets

## Summary

This dataset supports query-focused structured evidence extraction from PubMed
titles and abstracts. Version 1 contains 60 query-PMID pairs across asthma, type
2 diabetes, statin prevention, melanoma immunotherapy, and dietary sodium
topics. Version 2 extends the same five topics to 120 pairs over 108 unique
PMIDs.

The tracked source consists of a 20-row pilot, a 40-row v1 expansion, and a
60-row v2 expansion. Labels are model-assisted drafts and preserve
`annotation_status=draft` explicitly.

## Schema and validation

Every row targets `ModelEvidenceExtraction` v1 and is checked for:

- valid JSON and Pydantic schema compliance;
- closed enum values and status-dependent nullability;
- exact verbatim outcome spans in the source abstract;
- unique annotation IDs and query-PMID pairs;
- deterministic PMID-level split ownership.

## Composition

### Version 1

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

### Version 2

| Evidence status | Rows |
| --- | ---: |
| direct | 20 |
| indirect | 65 |
| none | 35 |

| Split | Rows | Unique PMIDs | direct | indirect | none |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 94 | 87 | 14 | 54 | 26 |
| dev | 13 | 11 | 3 | 7 | 3 |
| test | 13 | 10 | 3 | 4 | 6 |

## Provenance

- Documents: PubMed titles and abstracts in the BioEvidence demo corpus.
- Pilot: model-assisted annotation with a second-model review.
- Expansion: produced in Claude Desktop, followed by repository schema,
  grounding, count, duplicate, and candidate-coverage validation.
- Split seed: 42.
- Split unit: PMID.
- Version 2 freezes every v1 PMID assignment before assigning unseen PMIDs.

The source annotations, metadata, and aggregate split manifest are tracked under
`data/evaluations/evidence_extraction/`. Generated chat-format training files
remain under ignored `artifacts/` directories.

## Intended use and limitations

Accepted uses are pipeline validation, structured-output fine-tuning, and
held-out comparisons that name this dataset version. The dataset is not a
clinical benchmark or expert-gold biomedical corpus.

Version 1 coverage is uneven: direct labels occur only in the diabetes, statin,
and sodium query groups. Version 2 adds direct examples for asthma and melanoma,
but all labels remain drafts and the 13-row test set still makes metric estimates
unstable.

## Version 2 expansion selection

The tracked `expansion_candidates.v2.jsonl` queue contains 60 query-PMID pairs
after excluding all 60 v1 annotations. It samples 40 same-topic high-ranking and
20 same-topic broad documents, with no additional cross-topic hard negatives.
The matching annotation file passed exact 60/60 candidate coverage and contains
12 direct, 36 indirect, and 12 none labels. Candidate selection targeted the
first adapter's conservative failure mode but did not prescribe those labels.
