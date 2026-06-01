param(
  [string]$PythonExe = ".env\Scripts\python.exe",
  [string]$DataRoot = "training_data"
)

$ErrorActionPreference = "Stop"

Write-Host "[1/4] Preparing training manifests and local dataset cache..."
& $PythonExe scripts/prepare_training_data.py --output_root $DataRoot

Write-Host "[2/4] Starting ASR fine-tuning (CPU-safe config)..."
& $PythonExe Qwen3-ASR/finetuning/qwen3_asr_sft.py `
  --model_path Qwen/Qwen3-ASR-0.6B `
  --train_file "$DataRoot/manifests/asr/train.jsonl" `
  --eval_file "$DataRoot/manifests/asr/eval.jsonl" `
  --output_dir "$DataRoot/outputs/asr" `
  --batch_size 1 `
  --grad_acc 8 `
  --lr 2e-5 `
  --epochs 1 `
  --save_steps 200 `
  --save_total_limit 2 `
  --num_workers 0 `
  --pin_memory 0 `
  --persistent_workers 0

Write-Host "[3/4] Fine-tuning NLLB for (vi/zh/ja) -> English..."
$toEnglishRuns = @(
  @{ Pair = "en_vi"; Source = "vie_Latn" },
  @{ Pair = "en_zh"; Source = "zho_Hans" },
  @{ Pair = "en_ja"; Source = "jpn_Jpan" }
)

foreach ($run in $toEnglishRuns) {
  Write-Host ("  - Training " + $run.Pair + " -> English")
  & $PythonExe -m translate_model.train_nllb `
    --train_file "$DataRoot/manifests/translation/$($run.Pair)_to_english_train.jsonl" `
    --eval_file "$DataRoot/manifests/translation/$($run.Pair)_to_english_eval.jsonl" `
    --output_dir "$DataRoot/outputs/nllb_$($run.Pair)_to_english" `
    --source_lang $run.Source `
    --target_lang eng_Latn `
    --num_train_epochs 1 `
    --per_device_train_batch_size 2 `
    --per_device_eval_batch_size 2 `
    --gradient_accumulation_steps 4 `
    --learning_rate 3e-5 `
    --save_steps 500 `
    --eval_steps 500
}

Write-Host "[4/4] Fine-tuning NLLB for English -> (vi/zh/ja)..."
$fromEnglishRuns = @(
  @{ Pair = "en_vi"; Target = "vie_Latn" },
  @{ Pair = "en_zh"; Target = "zho_Hans" },
  @{ Pair = "en_ja"; Target = "jpn_Jpan" }
)

foreach ($run in $fromEnglishRuns) {
  Write-Host ("  - Training English -> " + $run.Pair)
  & $PythonExe -m translate_model.train_nllb `
    --train_file "$DataRoot/manifests/translation/english_to_$($run.Pair)_train.jsonl" `
    --eval_file "$DataRoot/manifests/translation/english_to_$($run.Pair)_eval.jsonl" `
    --output_dir "$DataRoot/outputs/nllb_english_to_$($run.Pair)" `
    --source_lang eng_Latn `
    --target_lang $run.Target `
    --num_train_epochs 1 `
    --per_device_train_batch_size 2 `
    --per_device_eval_batch_size 2 `
    --gradient_accumulation_steps 4 `
    --learning_rate 3e-5 `
    --save_steps 500 `
    --eval_steps 500
}

Write-Host "Pipeline completed. Outputs are under $DataRoot/outputs"
