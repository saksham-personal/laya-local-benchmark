# Windows VDI runbook

Target: CPU-only Windows VDI, 8 vCPU Xeon Gold 6240R, GitHub accessible, Hugging Face blocked. Run in PowerShell. Use a volume with at least several GiB free for the 778 MB model ZIP, extracted weights, Python runtime, and local ONNX exports. The exact CPU and memory actually observed are written to the result directory.

## 1. Clone and verify assets

    gh auth status
    gh repo clone saksham-personal/laya-local-benchmark
    cd laya-local-benchmark
    powershell -ExecutionPolicy Bypass -File scripts/setup_offline.ps1 -WorkDir C:\LayaBench

The script downloads the pinned Python runtime from the company-reranker-benchmark GitHub Release, then the Laya model and exporter wheels from this repository's assets-v1 Release. It checks archive and extracted-file SHA256 values. It installs Python 3.12 wheels without reaching PyPI. It makes no Hugging Face request.

For a completely disconnected VDI, use a separate connected computer to transfer all three release ZIPs into C:\LayaBench\downloads; setup then consumes these local copies. The exact names and hashes are in configs/assets.json. To check an archive manually:

    Get-FileHash C:\LayaBench\downloads\laya-english-pytorch.zip -Algorithm SHA256

The expected English archive hash is d7c1d4819f13ca5c74f3c2191e18db4d403636bad590fbcccb7be2bad6ebe132. Do not accept a different archive.

## 2. Smoke and export

    C:\LayaBench\.venv\Scripts\python.exe scripts/run_quick.py --work-dir C:\LayaBench --results C:\LayaBench\quick
    C:\LayaBench\.venv\Scripts\python.exe scripts/export_onnx.py --model-dir C:\LayaBench\models\english --mode fp32
    C:\LayaBench\.venv\Scripts\python.exe scripts/run_quick.py --work-dir C:\LayaBench --results C:\LayaBench\fp32-check
    C:\LayaBench\.venv\Scripts\python.exe scripts/export_onnx.py --model-dir C:\LayaBench\models\english --mode int8

The export must be done on a machine with sufficient free memory. It uses the pinned upstream exporter and local checkpoint. If an export fails, retain the error log and continue with the working PyTorch variant. The upstream INT8 exporter is known to cause prediction drift; do not deploy it solely because it is smaller or faster.

## 3. Full measured run

    C:\LayaBench\.venv\Scripts\python.exe scripts/run_full.py --work-dir C:\LayaBench --results C:\LayaBench\full

This runs all 384 custom and 120 public cases for each available artifact. Only variants passing the held-out custom comparison proceed to latency measurement. Each passing variant gets a brief 1/4/8-thread sweep at 128 tokens; the best thread count by batch-1 p95 receives the full 128/256/512-token, batch-1/8 measurement. The sweep is sequential; do not run two benchmark processes at once. Keep other CPU-heavy tasks closed. Timings use a warm filesystem cache, report sample counts, and are not durable capacity planning until repeated under realistic VDI load.

Outputs: C:\LayaBench\full\raw contains per-row predictions and per-variant summaries; machine.json records the host; the suite JSON records gates and file paths; reports contains report.md, report.html, report.json, and CSV quality, calibration, selective-routing, and latency tables. Share those artifacts when reviewing results. The generated report will explicitly flag a host that is not identified as the target VDI.

## 4. Decision rule

Choose among artifacts only after checking per-task company recall, false escalation, calibration, selective error, runtime memory, and latency at the four-thread allocation. INT8 must pass both reference fidelity and company quality gates. The frozen data is a pilot; collect representative, human-judged requests from the actual company workflow before production selection.
