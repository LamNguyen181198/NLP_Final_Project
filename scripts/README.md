# Scripts

This folder contains the small utility scripts used to prepare, preload, and train the project locally.

## Files

- `preload_models.py`: preloads ASR and translation models into the local cache.
- `prepare_training_data.py`: builds ASR and translation manifests under `training_data/`.
- `run_training_pipeline.ps1`: runs the full local training pipeline end to end.
- `install_and_preload.ps1`: installs dependencies and preloads models in one step.

## Common usage

From the workspace root:

```powershell
python scripts/preload_models.py
```

```powershell
python scripts/prepare_training_data.py --output_root training_data
```

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_training_pipeline.ps1
```

## Notes

- These scripts assume the workspace root layout used by the rest of the project.
- The training-related scripts write large outputs into `training_data/`.