# Third-party sources

- Laya: https://github.com/NandhaKishorM/laya, commit 8a6e1328cce2460a0e5aa348ad465bb1b5821cd2, Apache-2.0. The vendored source is in vendor/laya and the license is in vendor/LICENSE-LAYA.txt. vendor/export_onnx_upstream.py is copied from the same commit. The local ONNXAgent patch reads LAYA_BENCH_ORT_THREADS and fixes ORT execution to sequential CPU scheduling for reproducible thread sweeps.
- English Laya checkpoint: https://huggingface.co/convaiinnovations/laya, revision 7b928d828b7b0e022f929d9bd2e44165aa270148, Apache-2.0. Only five required root checkpoint/config/tokenizer files are packaged, each with SHA256 in configs/model-english.asset.json.
- BFCL: https://github.com/EnlightenedAI/BFCL, commit 6ea57973c7a6097fd7c5915698c54c17c5b1b6c8, Apache-2.0 data. The 120-row strict derivative and its transformation are documented in reports/PUBLIC_DATASET.md.
- Windows Python runtime: existing checksum-pinned bundle from https://github.com/saksham-personal/company-reranker-benchmark/releases/tag/assets-v1. Its original per-file checksums are preserved in configs/assets.json. Extra ONNX export wheels are packaged separately with exact wheel hashes.

The benchmark's original synthetic company cases and evaluation code are distinct from the upstream model and BFCL data.
