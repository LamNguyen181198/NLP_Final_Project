# NLP Final Project

This workspace is split into a few self-contained parts:

- [frontend](frontend/README.md): Flask API and web UI for ASR plus translation.
- [translate_model](translate_model/README.md): multilingual NLLB training and inference.
- [Qwen3-ASR](Qwen3-ASR/README.md): upstream ASR package and fine-tuning helpers.
- [scripts](scripts/README.md): local setup, preload, and training orchestration scripts.
- [training_data](training_data/README.md): generated manifests, cached data, and training outputs.

## Start Here

If you only want to run the app, begin with [frontend/README.md](frontend/README.md). That file contains the runtime steps, model preload instructions, and API examples.

If you want to train or regenerate data, use [TRAINING_QUICKSTART.md](TRAINING_QUICKSTART.md) together with [scripts/README.md](scripts/README.md) and [training_data/README.md](training_data/README.md).

## Project Layout

- Root: workspace-level overview and links to each part of the project.
- Frontend: user-facing Flask app.
- Translation: reusable translation model code and training entrypoints.
- ASR: upstream speech recognition package and its finetuning code.
- Scripts: repeatable helpers for preload and pipeline execution.
- Training data: large generated artifacts kept outside the main source tree.

## Notes

- Large generated assets live under `training_data/` and are intentionally separated from source code.
- Folder-specific setup details live in the README inside each folder.
