# Training Quickstart (ASR + Translation)

This project now supports a separate local folder for large training assets.

## 1) Prepare data into separate folder

```powershell
.env\Scripts\python.exe scripts/prepare_training_data.py --output_root training_data
```

This writes:

- `training_data/manifests/asr/train.jsonl`
- `training_data/manifests/asr/eval.jsonl`
- `training_data/manifests/translation/*.jsonl`
- `training_data/manifests/summary.json`

Large dataset cache is stored under `training_data/hf_cache`.

## 2) Train ASR (Qwen3-ASR)

```powershell
.env\Scripts\python.exe Qwen3-ASR/finetuning/qwen3_asr_sft.py `
  --model_path Qwen/Qwen3-ASR-0.6B `
  --train_file training_data/manifests/asr/train.jsonl `
  --eval_file training_data/manifests/asr/eval.jsonl `
  --output_dir training_data/outputs/asr `
  --batch_size 1 --grad_acc 8 --lr 2e-5 --epochs 1 `
  --num_workers 0 --pin_memory 0 --persistent_workers 0
```

## 3) Train translation models (NLLB)

Example: Vietnamese to English

```powershell
.env\Scripts\python.exe -m translate_model.train_nllb `
  --train_file training_data/manifests/translation/en_vi_to_english_train.jsonl `
  --eval_file training_data/manifests/translation/en_vi_to_english_eval.jsonl `
  --output_dir training_data/outputs/nllb_en_vi_to_english `
  --source_lang vie_Latn --target_lang eng_Latn `
  --num_train_epochs 1 --per_device_train_batch_size 2 --per_device_eval_batch_size 2 `
  --gradient_accumulation_steps 4 --learning_rate 3e-5
```

For all configured directions in one run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_training_pipeline.ps1
```

## Notes

- `training_data/` is ignored by git.
- ASR fine-tuning script has been patched for CPU-only fallback (`float32`) when CUDA is unavailable.
- On CPU, start with small epochs and smaller sample sizes in `prepare_training_data.py` arguments.
