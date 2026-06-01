# Frontend ASR API (Flask + Swagger)

This service exposes a Flask API that accepts an audio file, transcribes it with Qwen3-ASR, and then translates the transcript with NLLB-200.

## What is included

- `POST /api/v1/transcribe`: upload one audio file and receive transcript text plus translation text.
- `GET /health`: health check.
- `GET /health/models`: verify ASR/translation model files are already cached locally.
- `POST /admin/warmup`: load ASR + translation models into memory before first transcription.
- Web UI at `/`.
- Swagger UI at `/apidocs`.

## Quick start

1. Create and activate your Python environment.
2. Install dependencies from workspace root:

```bash
pip install -r requirements.txt
```

3. Configure environment values in the root `.env` file.
  The app now auto-loads `../.env` when started from the `frontend` folder.

4. Preload model files once (recommended):

```bash
python ../scripts/preload_models.py
```

On Windows, this preload path uses Transformers cache loading (not Hub snapshot symlinks), so it avoids the common `WinError 1314` symlink privilege issue.

Or run install + preload in one shot from workspace root:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/install_and_preload.ps1
```

5. Optional: after preload finishes, set `MODEL_LOCAL_ONLY=true` to prevent network fetch attempts on restart.

6. Run the app:

```bash
python app.py
```

7. Verify model cache status:

```bash
curl http://localhost:5000/health/models
```

Expected `status` is `ready` when both configured models are cached.

8. Optional: warm model weights into RAM to reduce first-request latency:

```bash
curl -X POST http://localhost:5000/admin/warmup
```

You can also open this directly in a browser:

- http://localhost:5000/admin/warmup

9. Open Swagger UI:

- http://localhost:5000/apidocs

10. Open web UI:

- http://localhost:5000/

## API example (curl)

```bash
curl -X POST "http://localhost:5000/api/v1/transcribe" \
  -F "file=@sample.wav" \
  -F "language=Vietnamese"
```

## Notes

- By default, the API uses `Qwen/Qwen3-ASR-0.6B` on CPU.
- First request may be slow because the model loads on demand.
- Model instance is cached and reused across requests.
- If you have a GPU, set `ASR_DEVICE_MAP=cuda:0` and a matching dtype.
- Translation defaults to `facebook/nllb-200-distilled-600M` and English output (`eng_Latn`).
- If you want to override the translation source language manually, set `TRANSLATION_SOURCE_LANG` to an NLLB code such as `deu_Latn` or `jpn_Jpan`.
- Downloaded model files are cached under `../.hf-cache`, so later restarts should reuse the same files instead of downloading again.
- If `GET /health/models` returns `missing_cache`, run `python ../scripts/preload_models.py` once and restart the app.
- Punctuation is enabled by default with a low-cost heuristic pass on translated text (`PUNCTUATION_STAGE=post_translation`, `PUNCTUATION_STRATEGY=heuristic`).
- For stronger punctuation restoration, set `PUNCTUATION_STRATEGY=model` (uses `deepmultilingualpunctuation`, slower than heuristic but still cheaper than a second LLM call).
- To reduce first transcription latency, either call `POST /admin/warmup` after app start or set `PRELOAD_MODELS_ON_STARTUP=true`.

## Docker

Build and run from workspace root (uses the same root `.env`):

```bash
docker compose up --build
```

Then open:

- http://localhost:5000/
- http://localhost:5000/apidocs
