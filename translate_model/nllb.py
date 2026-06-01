import argparse
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Optional


def _configure_default_cache() -> None:
    root_dir = Path(__file__).resolve().parents[1]
    cache_dir = root_dir / ".hf-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    transformers_cache_dir = cache_dir / "transformers"
    transformers_cache_dir.mkdir(parents=True, exist_ok=True)
    hub_cache_dir = cache_dir / "hub"
    hub_cache_dir.mkdir(parents=True, exist_ok=True)
    datasets_cache_dir = cache_dir / "datasets"
    datasets_cache_dir.mkdir(parents=True, exist_ok=True)
    torch_cache_dir = cache_dir / "torch"
    torch_cache_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("HF_HOME", str(cache_dir))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(hub_cache_dir))
    os.environ.setdefault("HF_DATASETS_CACHE", str(datasets_cache_dir))
    os.environ.setdefault("TORCH_HOME", str(torch_cache_dir))
    if os.getenv("MODEL_LOCAL_ONLY", "false").lower() == "true":
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


_configure_default_cache()

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

DEFAULT_MODEL_ID = "facebook/nllb-200-distilled-600M"

NLLB_LANGUAGE_CODES: Dict[str, str] = {
    "Arabic": "arb_Arab",
    "Chinese": "zho_Hans",
    "Cantonese": "yue_Hant",
    "Czech": "ces_Latn",
    "Danish": "dan_Latn",
    "Dutch": "nld_Latn",
    "English": "eng_Latn",
    "Finnish": "fin_Latn",
    "French": "fra_Latn",
    "German": "deu_Latn",
    "Greek": "ell_Grek",
    "Hindi": "hin_Deva",
    "Hungarian": "hun_Latn",
    "Indonesian": "ind_Latn",
    "Italian": "ita_Latn",
    "Japanese": "jpn_Jpan",
    "Korean": "kor_Hang",
    "Macedonian": "mkd_Cyrl",
    "Malay": "zsm_Latn",
    "Persian": "pes_Arab",
    "Filipino": "tgl_Latn",
    "Polish": "pol_Latn",
    "Portuguese": "por_Latn",
    "Romanian": "ron_Latn",
    "Russian": "rus_Cyrl",
    "Spanish": "spa_Latn",
    "Swedish": "swe_Latn",
    "Thai": "tha_Thai",
    "Turkish": "tur_Latn",
    "Vietnamese": "vie_Latn",
}


def normalize_language_code(language: Optional[str]) -> Optional[str]:
    if language is None:
        return None

    cleaned = str(language).strip()
    if not cleaned:
        return None

    if cleaned in NLLB_LANGUAGE_CODES.values():
        return cleaned

    canonical = cleaned[:1].upper() + cleaned[1:].lower()
    return NLLB_LANGUAGE_CODES.get(canonical, cleaned)


def resolve_source_language(asr_language: Optional[str], explicit_source_lang: Optional[str] = None) -> Optional[str]:
    explicit = normalize_language_code(explicit_source_lang)
    if explicit:
        return explicit
    return normalize_language_code(asr_language)


def _forced_bos_token_id(tokenizer, target_lang: str) -> int:
    lang_code_to_id = getattr(tokenizer, "lang_code_to_id", None)
    if isinstance(lang_code_to_id, dict) and target_lang in lang_code_to_id:
        return lang_code_to_id[target_lang]

    token_id = tokenizer.convert_tokens_to_ids(target_lang)
    if token_id is None or token_id < 0:
        raise ValueError(f"Unknown NLLB target language code: {target_lang}")
    return token_id


@dataclass
class NLLBTranslationConfig:
    model_id: str = DEFAULT_MODEL_ID
    source_lang: Optional[str] = None
    target_lang: str = "eng_Latn"


class NLLBTranslator:
    def __init__(self, model_id: str = DEFAULT_MODEL_ID, source_lang: Optional[str] = None, target_lang: str = "eng_Latn"):
        cache_dir = str((Path(os.environ.get("HF_HOME", str(Path(__file__).resolve().parents[1] / ".hf-cache"))) / "transformers").resolve())
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_id, cache_dir=cache_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.default_source_lang = normalize_language_code(source_lang)
        self.default_target_lang = normalize_language_code(target_lang) or "eng_Latn"

    @torch.no_grad()
    def translate(
        self,
        text: str,
        source_lang: Optional[str] = None,
        target_lang: Optional[str] = None,
        max_new_tokens: int = 128,
    ) -> str:
        source = normalize_language_code(source_lang) or self.default_source_lang
        target = normalize_language_code(target_lang) or self.default_target_lang

        if not source:
            raise ValueError("source_lang is required for NLLB translation")
        if not target:
            raise ValueError("target_lang is required for NLLB translation")

        self.tokenizer.src_lang = source
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=256,
        ).to(self.device)

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=4,
            forced_bos_token_id=_forced_bos_token_id(self.tokenizer, target),
        )
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)


def build_argument_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--model_id", default=DEFAULT_MODEL_ID, help="Base or fine-tuned NLLB model id")
    parser.add_argument("--source_lang", default="eng_Latn", help="NLLB source language code, e.g. deu_Latn or jpn_Jpan")
    parser.add_argument("--target_lang", default="eng_Latn", help="NLLB target language code, e.g. eng_Latn or deu_Latn")
    parser.add_argument("--text", default=None, help="Text to translate")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    return parser