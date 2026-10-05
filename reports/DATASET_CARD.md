# Evaluation data card

## Intended use

This kit tests whether Laya can make small, typed orchestration decisions for a company-screening assistant. It is for model selection and failure analysis, not automatic approval of company exclusions, purchases, or production deployment.

## Datasets

| Set | Rows | Source | Gold | Split | What it measures |
|---|---:|---|---|---|---|
| Company routing | 384 | Authored synthetic counterfactual cases | Authored with each case | Pair-family dev/test split | Six local routing decisions |
| BFCL V4 Live tool choice | 120 | Pinned official BFCL multiple-function data | One official function name | Held-out test | Choice among function descriptions |

See CUSTOM_DATASET.md and PUBLIC_DATASET.md for source hashes and regeneration commands.

## Limitations

The custom set is synthetic, repeats templates, and uses fictional companies. Labels encode the intended product policy; they do not establish that the model can verify factual exclusions from a real company record. The public slice is a nearby tool-selection task and cannot be quoted as an official BFCL score. Neither set represents the user's full 120,000-row company corpus or distribution of live questions. Original BFCL candidate functions and text may be subject to dataset-specific use conditions; review the pinned source license before redistribution outside this benchmark.

Confidence and selective-routing thresholds are fitted only on custom development rows. They must be revalidated on representative company requests. An abstention decision should preserve access to a stronger model or human review, and any exclusion should be checked against cited company evidence.

## Evaluation protocol

Run the same frozen rows, question schema, and maximum input length for all artifacts. Report held-out per-task accuracy, macro F1, class recall and confusion matrices. Report binary ROC AUC and Brier when probabilities have both labels, plus selected-answer ECE. For selective routing, derive confidence thresholds on development rows and report achieved test coverage and accepted error. On the public set, report quality only; there is no public-set threshold fitting.

Artifact comparisons use matched row IDs and require both overall and per-task fidelity. Quality gates are predeclared in scripts/run_suite.py. Passing a synthetic gate does not establish calibration or safety for real decisions.
