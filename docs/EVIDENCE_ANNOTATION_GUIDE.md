# Evidence extraction annotation guide

## Purpose

This guide defines the v1 query-focused extraction task used to compare the
deterministic extractor, a prompted base model, and a future fine-tuned model.
One annotation pairs one user query with one PubMed title and abstract. The
target contains only fields that require semantic interpretation; PMID, title,
year, journal, and retrieval score remain deterministic workflow metadata.

The canonical runtime contract is `ModelEvidenceExtraction` in
`src/bioevidence/schemas/model_evidence.py`. Its exported JSON Schema is
`schemas/model_evidence_extraction.v1.schema.json`.

## Annotation workflow

1. Read the query before reading the title and abstract.
2. Assign `evidence_status` relative to that query.
3. Assign the document-level `study_design` from explicit methods language.
4. Fill only query-relevant population, intervention or exposure, comparator,
   and outcomes that are supported by the abstract.
5. Copy every `evidence_span` verbatim from the abstract.
6. Validate the complete row and record the review path before changing
   `annotation_status` from `draft` to `reviewed`.

`annotation_status` describes label stability, not whether the reviewer was a
person or a model. Draft labels may be used for schema development and pilot
training; `reviewed` marks labels frozen for a named evaluation version. Record
label source, review method, source corpus, and accepted uses in the dataset
metadata rather than inferring authority from the status name.

Generate a local Markdown review packet with full abstracts and checklists:

```powershell
python scripts/render_extraction_review.py
```

The report is written under `artifacts/annotation_reviews/`, which is ignored by
Git. Record accepted changes in the source JSONL rather than editing the report.

## Evidence status

| Value | Use when |
|---|---|
| `direct` | The abstract reports an observed or synthesized result that directly addresses the query relationship. |
| `indirect` | The abstract provides adjacent mechanism, protocol, biomarker, safety, adherence, or related-intervention evidence but does not directly answer the query. |
| `none` | The abstract has no substantive evidence relationship to the query. Topic-word overlap alone is insufficient. |
| `unclear` | The title and abstract are too incomplete or ambiguous to decide reliably. |

`direct` requires at least one grounded outcome. `direct` and `indirect` require
an evidence summary. For `none`, all query-focused content fields must be null
or empty, while `study_design` should still describe the document itself.

## Study design

Use the most specific design explicitly supported by the abstract:

- `randomized_controlled_trial`
- `non_randomized_interventional`
- `cohort`
- `case_control`
- `cross_sectional`
- `case_report_or_series`
- `systematic_review_or_meta_analysis`
- `narrative_review`
- `study_protocol`
- `preclinical_in_vivo`
- `in_vitro`
- `other`
- `not_reported`

Apply these precedence rules:

- A protocol without completed results is `study_protocol`, even when it
  describes a future randomized trial.
- A systematic search with quantitative or structured evidence synthesis is
  `systematic_review_or_meta_analysis`.
- An explicitly described narrative or expert review is `narrative_review`.
- A randomized crossover study is `randomized_controlled_trial`.
- Do not infer randomization, prospective design, or a comparator that the
  abstract does not report.

For an evidence or effect query, a study protocol with no completed results is
`none`, while its document-level design remains `study_protocol`. A result is
`direct` only when it addresses the core query relationship, including the
relevant intervention or exposure and outcome; matching only part of a compact
retrieval query is `indirect`.

## Extracted fields

### Population or system

Record the human population, animal model, cell line, or other studied system.
Use a concise normalized description rather than copying all eligibility
criteria. Use null when it is not reported or not query-relevant.

### Intervention or exposure

Record the treatment, exposure, biomarker, genotype, or factor evaluated by the
document. For observational evidence, this field may describe an exposure
rather than an assigned intervention.

### Comparator

Record an explicit control or comparison condition. Do not invent "placebo" or
"standard care" when the abstract does not state it. Use null when absent.

### Outcomes

Each outcome contains:

- `name`: concise normalized outcome name;
- `direction`: what happened to the measured outcome;
- `result_text`: a short normalized description, or null if the abstract does
  not support a reliable normalization;
- `evidence_span`: exact supporting text copied from the abstract.

Allowed directions are:

| Value | Meaning |
|---|---|
| `increased` | The measured value, frequency, odds, risk, or response increased. |
| `decreased` | The measured value, frequency, odds, or risk decreased. |
| `no_clear_difference` | No statistically or clinically clear difference was reported. |
| `mixed` | Results differed materially across outcomes, groups, or analyses. |
| `association_only` | An association was reported without a meaningful increase/decrease interpretation. |
| `not_reported` | The outcome is named, but its direction is not reported. |

Direction describes the observation, not whether it is medically beneficial.
For example, increased adverse-event incidence is `increased`, not "negative."

### Evidence summary

Write one query-focused sentence supported only by the title and abstract.
State when the evidence is indirect. Do not add clinical recommendations,
causal language beyond the design, or information from outside the abstract.

## Evidence-span rules

- Copy the smallest complete phrase or sentence that supports the outcome.
- Preserve spelling, punctuation, symbols, and Unicode characters exactly.
- Do not repair grammar or expand abbreviations inside the span.
- Do not combine non-contiguous passages into one span.
- If no verbatim support exists, omit the outcome rather than fabricate a span.

The dataset loader rejects any span that is not an exact substring of the
corresponding tracked abstract.

## Dataset splitting and review

- Split train, dev, and test data by PMID, not by individual annotation row.
- Keep query variants for the same PMID in one split.
- Track source corpus version and annotation status.
- Freeze the label version and review provenance used for final metrics.
- Resolve disagreements through the written rules and record schema changes in
  `docs/DECISIONS.md`.

The tracked pilot intentionally contains direct, indirect, and negative pairs.
It supports schema pressure-testing, pipeline validation, and a pilot SFT run;
its dataset metadata records how the labels were produced and reviewed.

## Expansion queue

Build the next annotation batch with:

```powershell
python scripts/build_extraction_candidates.py
```

The deterministic queue excludes existing query-PMID pairs and samples three
bands per query: high-scoring topic documents, broader topic coverage, and
cross-topic hard negatives. Its manifest pins the source-corpus hash and
selection counts. A full prompt queue using the runtime extraction prompt is
written under ignored `artifacts/`.

Validate the configured model-assisted drafting job without making requests:

```powershell
python scripts/draft_extraction_candidates.py --dry-run
```

Running the command without `--dry-run` uses `EXTRACTION_*` configuration, or
falls back to the existing `AGENT_*` OpenAI-compatible endpoint. Only outputs
that pass JSON parsing, the Pydantic contract, and exact span grounding enter
the draft JSONL; raw failures remain separately inspectable.

The v2 queue excludes both tracked v1 annotation files and intentionally
selects only same-topic high and broad documents. The first adapter classified
all 17 training `none` rows correctly but collapsed many `direct` and
`indirect` rows toward lower evidence status, so this queue does not add more
cross-topic hard negatives. See `docs/EXTRACTION_ANNOTATION_TASK_V2.md` for the
exact input, output, and validation contract.
