import os
import re
import tempfile
import threading
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")
load_dotenv(Path(__file__).resolve().parent / ".env")

# Keep model downloads in a stable cache folder so restarts reuse them.
HF_CACHE_DIR = ROOT_DIR / ".hf-cache"
HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(HF_CACHE_DIR / "hub"))
os.environ.setdefault("HF_DATASETS_CACHE", str(HF_CACHE_DIR / "datasets"))
os.environ.setdefault("TORCH_HOME", str(HF_CACHE_DIR / "torch"))
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
if os.getenv("MODEL_LOCAL_ONLY", "false").lower() == "true":
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# Suppress known noisy warnings that do not affect inference correctness.
warnings.filterwarnings("ignore", message=r".*TRANSFORMERS_CACHE.*deprecated.*")
warnings.filterwarnings("ignore", message=r".*generation flags are not valid.*")
warnings.filterwarnings("ignore", message=r".*Setting `pad_token_id` to `eos_token_id`.*")
warnings.filterwarnings("ignore", message=r".*TripleDES.*")
warnings.filterwarnings("ignore", message=r".*Blowfish.*")

from flask import Flask, jsonify, render_template, request
from flasgger import Swagger, swag_from

import sys

# Optional local source import: this lets the API use the sibling repo without pip install -e .
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

LOCAL_QWEN_REPO = ROOT_DIR / "Qwen3-ASR"
if LOCAL_QWEN_REPO.exists():
    sys.path.insert(0, str(LOCAL_QWEN_REPO))

app = Flask(__name__)
app.config["SWAGGER"] = {
    "title": "Frontend ASR API",
    "uiversion": 3,
}
swagger = Swagger(app)


ALLOWED_EXTENSIONS = {"wav", "mp3", "m4a", "flac", "ogg", "aac", "webm"}


@dataclass
class ASRConfig:
    model_name: str
    device_map: str
    dtype: str
    max_new_tokens: int


@dataclass
class TranslationConfig:
    model_path: str
    source_lang: Optional[str]
    target_lang: str


@dataclass
class PunctuationConfig:
    enabled: bool
    stage: str
    strategy: str


class ASRService:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._model: Optional[object] = None
        self._config: Optional[ASRConfig] = None

    @staticmethod
    def _parse_dtype(dtype_name: str):
        import torch

        d = dtype_name.strip().lower()
        if d in {"float32", "fp32"}:
            return torch.float32
        if d in {"float16", "fp16", "half"}:
            return torch.float16
        if d in {"bfloat16", "bf16"}:
            return torch.bfloat16
        raise ValueError("dtype must be one of: float32, float16, bfloat16")

    def _build_model(self, config: ASRConfig):
        from qwen_asr import Qwen3ASRModel

        dtype = self._parse_dtype(config.dtype)
        return Qwen3ASRModel.from_pretrained(
            config.model_name,
            device_map=config.device_map,
            dtype=dtype,
            max_new_tokens=config.max_new_tokens,
        )

    def get_model(self, config: ASRConfig):
        with self._lock:
            if self._model is None or self._config != config:
                self._model = self._build_model(config)
                self._config = config
            return self._model


class TranslationService:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._translator: Optional[object] = None
        self._config: Optional[TranslationConfig] = None

    def _build_translator(self, config: TranslationConfig):
        from translate_model.nllb import NLLBTranslator

        return NLLBTranslator(
            model_id=config.model_path,
            source_lang=config.source_lang,
            target_lang=config.target_lang,
        )

    def get_translator(self, config: TranslationConfig):
        with self._lock:
            if self._translator is None or self._config != config:
                self._translator = self._build_translator(config)
                self._config = config
            return self._translator


class PunctuationService:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._model = None

    def _heuristic_punctuate(self, text: str) -> str:
        cleaned = " ".join((text or "").strip().split())
        if not cleaned:
            return ""

        # Cheap, deterministic punctuation normalization for translation input/output.
        cleaned = cleaned[0].upper() + cleaned[1:]
        if cleaned[-1] not in {".", "!", "?", "…"}:
            cleaned += "."
        return cleaned

    def _get_model(self):
        with self._lock:
            if self._model is None:
                from deepmultilingualpunctuation import PunctuationModel

                self._model = PunctuationModel()
            return self._model

    def punctuate(self, text: str, strategy: str) -> str:
        if strategy == "heuristic":
            return self._heuristic_punctuate(text)

        if strategy == "model":
            model = self._get_model()
            # deepmultilingualpunctuation expects plain text and returns punctuated text.
            return model.restore_punctuation(text)

        raise ValueError("PUNCTUATION_STRATEGY must be one of: heuristic, model")


asr_service = ASRService()
translation_service = TranslationService()
punctuation_service = PunctuationService()


def _get_env_config() -> ASRConfig:
    return ASRConfig(
        model_name=os.getenv("ASR_MODEL_NAME", "Qwen/Qwen3-ASR-0.6B"),
        device_map=os.getenv("ASR_DEVICE_MAP", "cpu"),
        dtype=os.getenv("ASR_DTYPE", "float32"),
        max_new_tokens=int(os.getenv("ASR_MAX_NEW_TOKENS", "256")),
    )


def _get_translation_env_config() -> TranslationConfig:
    return TranslationConfig(
        model_path=os.getenv("TRANSLATION_MODEL_PATH", "facebook/nllb-200-distilled-600M"),
        source_lang=os.getenv("TRANSLATION_SOURCE_LANG", "").strip() or None,
        target_lang=os.getenv("TRANSLATION_TARGET_LANG", "eng_Latn"),
    )


def _get_punctuation_env_config() -> PunctuationConfig:
    enabled = os.getenv("ENABLE_PUNCTUATION", "true").lower() == "true"
    stage = os.getenv("PUNCTUATION_STAGE", "post_translation").strip().lower()
    strategy = os.getenv("PUNCTUATION_STRATEGY", "heuristic").strip().lower()
    return PunctuationConfig(enabled=enabled, stage=stage, strategy=strategy)


def _resolve_translation_source_language(asr_language: Optional[str], config: TranslationConfig) -> Optional[str]:
    if config.source_lang:
        return config.source_lang

    from translate_model.nllb import resolve_source_language

    return resolve_source_language(asr_language)


def _translate_text(translator, text: str, source_lang: Optional[str], target_lang: str) -> str:
    cleaned = (text or "").strip()
    if not cleaned:
        return ""

    segments = [segment.strip() for segment in re.split(r"(?<=[。！？!?])", cleaned) if segment.strip()]
    if len(segments) <= 1:
        return translator.translate(
            cleaned,
            source_lang=source_lang or None,
            target_lang=target_lang,
        )

    translated_segments = [
        translator.translate(
            segment,
            source_lang=source_lang or None,
            target_lang=target_lang,
        )
        for segment in segments
    ]
    return "\n".join(segment.strip() for segment in translated_segments if segment.strip())


def _is_allowed_file(filename: str) -> bool:
    if "." not in filename:
        return False
    return filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _check_model_cached(model_ref: str) -> dict:
    model_value = (model_ref or "").strip()
    if not model_value:
        return {
            "model": model_ref,
            "cached": False,
            "reason": "Model reference is empty",
        }

    maybe_local = Path(model_value)
    if maybe_local.exists():
        return {
            "model": model_value,
            "cached": True,
            "source": "local_path",
            "path": str(maybe_local.resolve()),
        }

    repo_dir_name = f"models--{model_value.replace('/', '--')}"
    hub_cache_dir = os.getenv("HUGGINGFACE_HUB_CACHE")
    transformers_cache_dir = os.getenv("TRANSFORMERS_CACHE") or str((HF_CACHE_DIR / "transformers").resolve())

    cache_candidates = []
    # Prefer transformers cache first on Windows because it avoids symlink privilege issues.
    if transformers_cache_dir:
        cache_candidates.append(Path(transformers_cache_dir) / repo_dir_name)
    if hub_cache_dir:
        cache_candidates.append(Path(hub_cache_dir) / repo_dir_name)

    for candidate in cache_candidates:
        if candidate.exists():
            return {
                "model": model_value,
                "cached": True,
                "source": "cache_folder",
                "path": str(candidate.resolve()),
                "cache_dir": str(candidate.parent.resolve()),
            }

    return {
        "model": model_value,
        "cached": False,
        "source": "cache_folder",
        "cache_dir": {
            "transformers": transformers_cache_dir,
            "hub": hub_cache_dir,
        },
        "reason": "Model folder not found in configured cache directories",
    }


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.get("/health/models")
@swag_from(
    {
        "tags": ["Health"],
        "summary": "Check whether ASR and translation models are already cached locally",
        "responses": {
            "200": {
                "description": "Cache status for configured models",
                "schema": {
                    "type": "object",
                    "properties": {
                        "status": {"type": "string"},
                        "cache_dir": {"type": "string"},
                        "model_local_only": {"type": "boolean"},
                        "models": {
                            "type": "object",
                            "properties": {
                                "asr": {"type": "object"},
                                "translation": {"type": "object"},
                            },
                        },
                    },
                },
            }
        },
    }
)
def health_models():
    asr_cfg = _get_env_config()
    tr_cfg = _get_translation_env_config()

    asr_status = _check_model_cached(asr_cfg.model_name)
    tr_status = _check_model_cached(tr_cfg.model_path)
    all_cached = bool(asr_status.get("cached") and tr_status.get("cached"))

    return jsonify(
        {
            "status": "ready" if all_cached else "missing_cache",
            "cache_dir": str(HF_CACHE_DIR.resolve()),
            "model_local_only": os.getenv("MODEL_LOCAL_ONLY", "false").lower() == "true",
            "models": {
                "asr": asr_status,
                "translation": tr_status,
            },
        }
    )


def _warmup_models() -> None:
    asr_cfg = _get_env_config()
    tr_cfg = _get_translation_env_config()
    asr_service.get_model(asr_cfg)
    translation_service.get_translator(tr_cfg)


@app.route("/admin/warmup", methods=["GET", "POST"])
@swag_from(
    {
        "tags": ["Admin"],
        "summary": "Load ASR and translation models into memory before first request",
        "description": "Accepts both GET (browser-friendly) and POST (API-friendly).",
        "responses": {
            "200": {
                "description": "Warmup completed",
                "schema": {
                    "type": "object",
                    "properties": {
                        "status": {"type": "string"}
                    },
                },
            },
            "500": {"description": "Warmup failed"},
        },
    }
)
def warmup_models():
    try:
        _warmup_models()
        return jsonify({"status": "warmed"})
    except Exception as exc:
        return jsonify({"status": "error", "error": str(exc)}), 500


@app.post("/api/v1/transcribe")
@swag_from(
    {
        "tags": ["ASR"],
        "summary": "Transcribe one audio file using Qwen3-ASR",
        "consumes": ["multipart/form-data"],
        "parameters": [
            {
                "name": "file",
                "in": "formData",
                "type": "file",
                "required": True,
                "description": "Audio file to transcribe",
            },
            {
                "name": "language",
                "in": "formData",
                "type": "string",
                "required": False,
                "description": "Optional forced language. Example: Vietnamese, English, Chinese",
            },
            {
                "name": "context",
                "in": "formData",
                "type": "string",
                "required": False,
                "description": "Optional context prompt",
            },
        ],
        "responses": {
            "200": {
                "description": "Transcription result",
                "schema": {
                    "type": "object",
                    "properties": {
                        "language": {"type": "string"},
                        "text": {"type": "string"},
                        "translated_text": {"type": "string"},
                        "translation_source_lang": {"type": "string"},
                        "translation_target_lang": {"type": "string"},
                        "translation_model": {"type": "string"},
                    },
                },
            },
            "400": {"description": "Bad request"},
            "500": {"description": "Server error"},
        },
    }
)
def transcribe_audio():
    file = request.files.get("file")
    if file is None or file.filename is None or file.filename.strip() == "":
        return jsonify({"error": "No file uploaded"}), 400

    if not _is_allowed_file(file.filename):
        return jsonify(
            {
                "error": "Unsupported file extension",
                "allowed_extensions": sorted(ALLOWED_EXTENSIONS),
            }
        ), 400

    context = (request.form.get("context") or "").strip()
    language = (request.form.get("language") or "").strip() or None

    env_cfg = _get_env_config()
    cfg = ASRConfig(
        model_name=env_cfg.model_name,
        device_map=env_cfg.device_map,
        dtype=env_cfg.dtype,
        max_new_tokens=env_cfg.max_new_tokens,
    )

    suffix = "." + file.filename.rsplit(".", 1)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        temp_path = tmp.name
        file.save(temp_path)

    try:
        model = asr_service.get_model(cfg)
        result = model.transcribe(
            audio=temp_path,
            context=context,
            language=language,
        )[0]

        translated_text = ""
        raw_translated_text = ""
        translation_model = ""
        translation_source_lang = ""
        translation_target_lang = ""
        translation_error = ""
        punctuation_error = ""
        punctuated_text = result.text
        punctuated_translated_text = ""
        translation_cfg = _get_translation_env_config()
        punctuation_cfg = _get_punctuation_env_config()

        text_for_translation = result.text
        if punctuation_cfg.enabled and punctuation_cfg.stage in {"pre_translation", "both"} and result.text.strip():
            try:
                punctuated_text = punctuation_service.punctuate(result.text, punctuation_cfg.strategy)
                text_for_translation = punctuated_text
            except Exception as exc:
                punctuation_error = f"pre_translation: {exc}"

        if result.text.strip():
            translator = translation_service.get_translator(translation_cfg)
            translation_source_lang = _resolve_translation_source_language(result.language, translation_cfg) or ""
            translation_target_lang = translation_cfg.target_lang
            try:
                raw_translated_text = _translate_text(
                    translator,
                    text_for_translation,
                    source_lang=translation_source_lang or None,
                    target_lang=translation_target_lang,
                )
                translated_text = raw_translated_text
                if punctuation_cfg.enabled and punctuation_cfg.stage in {"post_translation", "both"} and translated_text.strip():
                    try:
                        punctuated_translated_text = punctuation_service.punctuate(translated_text, punctuation_cfg.strategy)
                        translated_text = punctuated_translated_text
                    except Exception as exc:
                        suffix = f"post_translation: {exc}"
                        punctuation_error = f"{punctuation_error}; {suffix}" if punctuation_error else suffix

                translation_model = translation_cfg.model_path
            except ValueError as exc:
                translation_error = str(exc)

        return jsonify(
            {
                "language": result.language,
                "text": punctuated_text if punctuation_cfg.enabled else result.text,
                "raw_text": result.text,
                "asr_model_name": cfg.model_name,
                "translated_text": translated_text,
                "raw_translated_text": raw_translated_text,
                "translation_source_lang": translation_source_lang,
                "translation_target_lang": translation_target_lang,
                "translation_error": translation_error,
                "translation_model": translation_model,
                "punctuation_enabled": punctuation_cfg.enabled,
                "punctuation_stage": punctuation_cfg.stage,
                "punctuation_strategy": punctuation_cfg.strategy,
                "punctuation_error": punctuation_error,
            }
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": f"Transcription failed: {exc}"}), 500
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass


if __name__ == "__main__":
    if os.getenv("PRELOAD_MODELS_ON_STARTUP", "false").lower() == "true":
        threading.Thread(target=_warmup_models, daemon=True).start()

    host = os.getenv("FLASK_HOST", "0.0.0.0")
    port = int(os.getenv("FLASK_PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    app.run(host=host, port=port, debug=debug)
