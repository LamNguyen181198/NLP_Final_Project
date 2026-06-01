import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv


def configure_cache(root_dir: Path) -> Path:
    cache_dir = root_dir / ".hf-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    hub_cache_dir = cache_dir / "hub"
    hub_cache_dir.mkdir(parents=True, exist_ok=True)
    transformers_cache_dir = cache_dir / "transformers"
    transformers_cache_dir.mkdir(parents=True, exist_ok=True)
    datasets_cache_dir = cache_dir / "datasets"
    datasets_cache_dir.mkdir(parents=True, exist_ok=True)
    torch_cache_dir = cache_dir / "torch"
    torch_cache_dir.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(cache_dir)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hub_cache_dir)
    os.environ["HF_DATASETS_CACHE"] = str(datasets_cache_dir)
    os.environ["TORCH_HOME"] = str(torch_cache_dir)
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    return cache_dir


def parse_args() -> argparse.Namespace:
    root_dir = Path(__file__).resolve().parents[1]
    load_dotenv(root_dir / ".env")

    parser = argparse.ArgumentParser(description="Preload ASR and translation models into local cache.")
    parser.add_argument("--asr-model", default=os.getenv("ASR_MODEL_NAME", "Qwen/Qwen3-ASR-0.6B"), help="ASR model id")
    parser.add_argument(
        "--translation-model",
        default=os.getenv("TRANSLATION_MODEL_PATH", "facebook/nllb-200-distilled-600M"),
        help="Translation model id",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root_dir = Path(__file__).resolve().parents[1]
    cache_dir = configure_cache(root_dir)
    local_qwen_repo = root_dir / "Qwen3-ASR"
    if local_qwen_repo.exists() and str(local_qwen_repo) not in sys.path:
        sys.path.insert(0, str(local_qwen_repo))

    print(f"Using cache directory: {cache_dir}")
    transformers_cache_dir = str((Path(os.environ["HF_HOME"]) / "transformers").resolve())

    print(f"Preloading ASR model via qwen_asr loader: {args.asr_model}")
    try:
        import torch
        from qwen_asr import Qwen3ASRModel
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing ASR preload dependency. Install with: pip install -r requirements.txt"
        ) from exc

    asr_model = Qwen3ASRModel.from_pretrained(
        args.asr_model,
        device_map="cpu",
        dtype=torch.float32,
        max_new_tokens=16,
    )
    # Release memory after cache warm-up.
    del asr_model

    print(f"Preloading translation model via Transformers cache: {args.translation_model}")
    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing translation preload dependency. Install with: pip install -r requirements.txt"
        ) from exc

    AutoTokenizer.from_pretrained(args.translation_model, cache_dir=transformers_cache_dir)
    AutoModelForSeq2SeqLM.from_pretrained(args.translation_model, cache_dir=transformers_cache_dir)

    print("Model preload complete. Future restarts should reuse local cache.")


if __name__ == "__main__":
    main()