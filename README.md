# Laya local decision benchmark

Offline CPU benchmark and deployment kit for English Laya on a Windows VDI that can reach GitHub but cannot reach Hugging Face. It evaluates routing decisions for company screening and an adjacent public tool-choice slice. Pinned code and datasets are in Git; model and Python assets are delivered as checksum-verified GitHub Release archives.

## Repository and assets

- Repository: https://github.com/saksham-personal/laya-local-benchmark
- Asset Release: https://github.com/saksham-personal/laya-local-benchmark/releases/tag/assets-v1
- English model: laya-english-pytorch.zip, SHA256 d7c1d4819f13ca5c74f3c2191e18db4d403636bad590fbcccb7be2bad6ebe132
- Offline Windows Python 3.12 runtime: https://github.com/saksham-personal/company-reranker-benchmark/releases/tag/assets-v1
- ONNX export wheels: laya-export-wheels-windows-x64.zip, SHA256 bf710ee7bf2c0a1384cbacd5faf55afdbc2c9ee80d17a630aab5547fbc1a89fa

The release links become live after the repository and assets are published. Every archive and extracted file is checked against configs/assets.json before use.

## Quick start on the VDI

From PowerShell with GitHub CLI installed and authenticated:

    gh repo clone saksham-personal/laya-local-benchmark
    cd laya-local-benchmark
    powershell -ExecutionPolicy Bypass -File scripts/setup_offline.ps1 -WorkDir C:\LayaBench
    C:\LayaBench\.venv\Scripts\python.exe scripts/run_quick.py --work-dir C:\LayaBench --results C:\LayaBench\quick

Setup downloads only from GitHub, verifies the offline Python archive and model/export wheels, and installs a locked local environment. On a VDI with no internet at all, transfer the three named ZIPs into C:\LayaBench\downloads first, then run setup.

For the complete quality-first run:

    C:\LayaBench\.venv\Scripts\python.exe scripts/export_onnx.py --model-dir C:\LayaBench\models\english --mode fp32
    C:\LayaBench\.venv\Scripts\python.exe scripts/run_quick.py --work-dir C:\LayaBench --results C:\LayaBench\fp32-check
    C:\LayaBench\.venv\Scripts\python.exe scripts/export_onnx.py --model-dir C:\LayaBench\models\english --mode int8
    C:\LayaBench\.venv\Scripts\python.exe scripts/run_full.py --work-dir C:\LayaBench --results C:\LayaBench\full

The INT8 exporter is included for measurement, not assumed acceptable: the pinned upstream exporter warns of substantial prediction drift. The full suite gates exported variants against FP32 on held-out company cases and skips latency for a variant that fails. The model archive contains PyTorch FP32; ONNX files are generated locally on the VDI and retained there.

## What is measured

- Quality: held-out accuracy, macro F1, per-class recall, confusion matrix, binary ROC AUC/Brier where applicable, selected-answer ECE, and a development-threshold selective routing curve.
- Artifact fidelity: matched prediction agreement, accuracy change, class recall change, and probability deltas.
- CPU performance: local cold load, first call, p50/p95/p99 latency, throughput, and sampled peak RSS at batch 1 and 8, token buckets 128/256/512, and 1/4/8 threads.
- Reproducibility: pinned upstream commits, model revision, source hashes, deterministic data, raw prediction rows, machine metadata, JSON/CSV/Markdown/HTML reports.

Results from this development PC are not target VDI measurements. A production model recommendation needs the full run on the actual VDI and representative, human-judged company requests. See reports/VDI_SETUP.md and reports/DATASET_CARD.md.
