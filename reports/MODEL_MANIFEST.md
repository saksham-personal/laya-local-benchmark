# Model and runtime manifest

The English Laya root checkpoint is pinned to https://huggingface.co/convaiinnovations/laya at revision 7b928d828b7b0e022f929d9bd2e44165aa270148. The five packaged files, byte sizes, and SHA256 values are recorded in configs/model-english.asset.json. The model archive SHA256 is d7c1d4819f13ca5c74f3c2191e18db4d403636bad590fbcccb7be2bad6ebe132.

The model archive contains model.safetensors, rl_agent_config.json, encoder/config.json, tokenizer/tokenizer.json, and tokenizer/tokenizer_config.json. It does not contain ONNX exports. The frozen upstream Laya exporter creates laya.onnx and laya.int8.onnx locally; generated files must be validated before comparison or use.

The runtime bundle SHA256 and 3,446 extracted-file checksums are in configs/assets.json. The separate Windows ONNX export wheel archive SHA256 is bf710ee7bf2c0a1384cbacd5faf55afdbc2c9ee80d17a630aab5547fbc1a89fa. Exact hashes for all four wheels are in configs/export-wheels.asset.json and configs/export-requirements.txt. Only verified local files are loaded at benchmark time.
