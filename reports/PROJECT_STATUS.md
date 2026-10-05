# Delivery and verification status

As of 2026-10-06, the repository and assets-v1 Release are published. GitHub reports the expected SHA256 digests and sizes for the 778,428,250-byte English checkpoint ZIP and 9,252,490-byte ONNX export wheel ZIP. The reused offline Python ZIP is available from the company-reranker-benchmark Release with the hash pinned in configs/assets.json.

Completed local checks:

- 19 Python tests passed, including dataset conversion, archive extraction guards, metrics, report generation, and artifact gates.
- Python source compilation and PowerShell setup-script parsing passed.
- The model ZIP matched its manifest SHA256 and passed the ZIP CRC check.
- A GitHub-only download and verified extraction of the exporter wheel ZIP passed.
- The entire Windows Python setup was rehearsed offline from the three local ZIPs. It installed the locked base environment, extracted and verified the English checkpoint, and installed the four hash-pinned ONNX exporter wheels.
- The installed environment can import the benchmark and upstream exporter command lines.

The target VDI was not available to this run. There are no target VDI model quality, ONNX fidelity, latency, thread-scaling, or memory results yet. A one-case model invocation on the development PC exited without a diagnostic result; it is not counted as validation. Run the quick and full suites on the target VDI using VDI_SETUP.md, then inspect the generated report and raw predictions. Do not infer a production recommendation from packaging checks.
