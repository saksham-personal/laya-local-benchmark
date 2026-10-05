# Company routing dataset

The frozen file at data/synthetic/company_routing.jsonl contains 384 authored cases: 64 each for intent, external research, expensive reasoning, search type, evidence sufficiency, and direct action versus escalation. Cases mirror the company screening workflow with local structured fields, semantic company search, saved evidence, and current external research.

Each task has eight counterfactual prompt pairs, rendered with four state wrappers. A pair and its variants stay in one split. Seed: 20261005. Half the pair families are development cases and half are held-out test cases. SHA256: 4b8cc8be6419a51e1145b8133ea95e9fcd4e6c7fb61c688dd5b7981bdb0a0a53.

Gold decisions were written with the prompt pairs; no model generated labels. This is synthetic development evidence. Wording patterns and fictional companies may be easier than live requests. A production decision needs separately labeled, representative company requests, especially ambiguous or costly escalation cases.

Regenerate with:

    python scripts/build_custom_dataset.py data/synthetic/company_routing.jsonl

Inspect per-task confusion matrices, class recall, calibration, and the retained-versus-escalated curve instead of relying on pooled accuracy.
