import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from datasets import Audio, load_dataset


LANGUAGE_SPECS = [
    {"name": "Vietnamese", "fleurs_config": "vi_vn", "opus_code": "vi", "nllb_code": "vie_Latn"},
    {"name": "Chinese", "fleurs_config": "cmn_hans_cn", "opus_code": "zh", "nllb_code": "zho_Hans"},
    {"name": "Japanese", "fleurs_config": "ja_jp", "opus_code": "ja", "nllb_code": "jpn_Jpan"},
]


@dataclass
class PrepStats:
    written: int = 0
    skipped: int = 0


def _write_jsonl(path: Path, records: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _ensure_hf_cache(output_root: Path) -> None:
    hf_root = output_root / "hf_cache"
    hub = hf_root / "hub"
    datasets_cache = hf_root / "datasets"
    os.environ["HF_HOME"] = str(hf_root.resolve())
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hub.resolve())
    os.environ["HF_DATASETS_CACHE"] = str(datasets_cache.resolve())
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    hf_root.mkdir(parents=True, exist_ok=True)
    hub.mkdir(parents=True, exist_ok=True)
    datasets_cache.mkdir(parents=True, exist_ok=True)


def _slice_split(dataset_name: str, config: str, split: str, limit: int, seed: int):
    ds = load_dataset(dataset_name, config, split=split)
    if len(ds) > limit:
        ds = ds.shuffle(seed=seed).select(range(limit))
    return ds


def _safe_slug(name: str) -> str:
    return name.lower().replace(" ", "_")


def _materialize_audio(sample: Dict[str, object], audio_dir: Path, index: int, split_name: str) -> str:
    audio = sample.get("audio") or {}
    audio_bytes = audio.get("bytes")
    if not audio_bytes:
        raise ValueError("Missing audio bytes")

    audio_dir.mkdir(parents=True, exist_ok=True)
    audio_path = audio_dir / f"{split_name}_{index:08d}.wav"
    with audio_path.open("wb") as handle:
        handle.write(audio_bytes)
    return str(audio_path.resolve())


def _build_asr_records(output_root: Path, limit_train: int, limit_eval: int, seed: int) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], Dict[str, PrepStats]]:
    train_records: List[Dict[str, str]] = []
    eval_records: List[Dict[str, str]] = []
    stats: Dict[str, PrepStats] = {}

    for spec in LANGUAGE_SPECS:
        lang_name = spec["name"]
        stats[lang_name] = PrepStats()

        train_ds = _slice_split(
            dataset_name="google/fleurs",
            config=spec["fleurs_config"],
            split="train",
            limit=limit_train,
            seed=seed,
        )
        eval_ds = _slice_split(
            dataset_name="google/fleurs",
            config=spec["fleurs_config"],
            split="test",
            limit=limit_eval,
            seed=seed,
        )
        train_ds = train_ds.cast_column("audio", Audio(decode=False))
        eval_ds = eval_ds.cast_column("audio", Audio(decode=False))
        train_audio_dir = output_root / "audio" / "asr" / _safe_slug(lang_name) / "train"
        eval_audio_dir = output_root / "audio" / "asr" / _safe_slug(lang_name) / "eval"

        for ds, sink, split_name, audio_dir in (
            (train_ds, train_records, "train", train_audio_dir),
            (eval_ds, eval_records, "eval", eval_audio_dir),
        ):
            for idx, sample in enumerate(ds):
                text = str(sample.get("sentence") or "").strip()
                if not text:
                    text = str(sample.get("transcription") or "").strip()
                if not text:
                    stats[lang_name].skipped += 1
                    continue
                try:
                    audio_path = _materialize_audio(sample, audio_dir=audio_dir, index=idx, split_name=split_name)
                except Exception:
                    stats[lang_name].skipped += 1
                    continue
                sink.append(
                    {
                        "audio": audio_path,
                        "text": f"language {lang_name}<asr_text>{text}",
                    }
                )
                stats[lang_name].written += 1

    return train_records, eval_records, stats


def _build_translation_records(limit_train: int, limit_eval: int, seed: int):
    to_en_train: List[Dict[str, str]] = []
    to_en_eval: List[Dict[str, str]] = []
    from_en_train: List[Dict[str, str]] = []
    from_en_eval: List[Dict[str, str]] = []
    per_pair_stats: Dict[str, Dict[str, int]] = {}
    per_pair_files: Dict[str, Dict[str, List[Dict[str, str]]]] = {}

    for spec in LANGUAGE_SPECS:
        pair_name = f"en-{spec['opus_code']}"
        per_pair_stats[pair_name] = {"written_train": 0, "written_eval": 0, "skipped": 0}
        per_pair_files[pair_name] = {
            "to_en_train": [],
            "to_en_eval": [],
            "from_en_train": [],
            "from_en_eval": [],
        }

        train_ds = _slice_split(
            dataset_name="Helsinki-NLP/opus-100",
            config=pair_name,
            split="train",
            limit=limit_train,
            seed=seed,
        )
        eval_ds = _slice_split(
            dataset_name="Helsinki-NLP/opus-100",
            config=pair_name,
            split="validation",
            limit=limit_eval,
            seed=seed,
        )

        for ds, to_sink, from_sink, per_pair_to_sink, per_pair_from_sink, split_key in (
            (train_ds, to_en_train, from_en_train, per_pair_files[pair_name]["to_en_train"], per_pair_files[pair_name]["from_en_train"], "written_train"),
            (eval_ds, to_en_eval, from_en_eval, per_pair_files[pair_name]["to_en_eval"], per_pair_files[pair_name]["from_en_eval"], "written_eval"),
        ):
            for sample in ds:
                trans = sample.get("translation") or {}
                en_text = str(trans.get("en") or "").strip()
                xx_text = str(trans.get(spec["opus_code"]) or "").strip()
                if not en_text or not xx_text:
                    per_pair_stats[pair_name]["skipped"] += 1
                    continue

                # xx -> en
                to_record = {
                    "source": xx_text,
                    "target": en_text,
                    "source_lang": spec["nllb_code"],
                    "target_lang": "eng_Latn",
                }
                to_sink.append(to_record)
                per_pair_to_sink.append(to_record)

                # en -> xx
                from_record = {
                    "source": en_text,
                    "target": xx_text,
                    "source_lang": "eng_Latn",
                    "target_lang": spec["nllb_code"],
                }
                from_sink.append(from_record)
                per_pair_from_sink.append(from_record)
                per_pair_stats[pair_name][split_key] += 1

    return to_en_train, to_en_eval, from_en_train, from_en_eval, per_pair_stats, per_pair_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare ASR and translation training data into a separate local folder.")
    parser.add_argument("--output_root", default="training_data", help="Root folder for large local training assets")
    parser.add_argument("--asr_train_per_lang", type=int, default=6000)
    parser.add_argument("--asr_eval_per_lang", type=int, default=800)
    parser.add_argument("--mt_train_per_pair", type=int, default=30000)
    parser.add_argument("--mt_eval_per_pair", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    _ensure_hf_cache(output_root)

    manifests_root = output_root / "manifests"
    asr_root = manifests_root / "asr"
    mt_root = manifests_root / "translation"

    asr_train, asr_eval, asr_stats = _build_asr_records(
        output_root=output_root,
        limit_train=args.asr_train_per_lang,
        limit_eval=args.asr_eval_per_lang,
        seed=args.seed,
    )
    _write_jsonl(asr_root / "train.jsonl", asr_train)
    _write_jsonl(asr_root / "eval.jsonl", asr_eval)

    mt_to_en_train, mt_to_en_eval, mt_from_en_train, mt_from_en_eval, mt_stats, mt_pair_files = _build_translation_records(
        limit_train=args.mt_train_per_pair,
        limit_eval=args.mt_eval_per_pair,
        seed=args.seed,
    )
    _write_jsonl(mt_root / "nllb_to_english_train.jsonl", mt_to_en_train)
    _write_jsonl(mt_root / "nllb_to_english_eval.jsonl", mt_to_en_eval)
    _write_jsonl(mt_root / "nllb_from_english_train.jsonl", mt_from_en_train)
    _write_jsonl(mt_root / "nllb_from_english_eval.jsonl", mt_from_en_eval)

    for pair_name, pair_files in mt_pair_files.items():
        slug = pair_name.replace("-", "_")
        _write_jsonl(mt_root / f"{slug}_to_english_train.jsonl", pair_files["to_en_train"])
        _write_jsonl(mt_root / f"{slug}_to_english_eval.jsonl", pair_files["to_en_eval"])
        _write_jsonl(mt_root / f"english_to_{slug}_train.jsonl", pair_files["from_en_train"])
        _write_jsonl(mt_root / f"english_to_{slug}_eval.jsonl", pair_files["from_en_eval"])

    summary = {
        "output_root": str(output_root),
        "hf_cache": str((output_root / "hf_cache").resolve()),
        "asr": {
            "train_file": str((asr_root / "train.jsonl").resolve()),
            "eval_file": str((asr_root / "eval.jsonl").resolve()),
            "train_rows": len(asr_train),
            "eval_rows": len(asr_eval),
            "language_stats": {
                k: {"written": v.written, "skipped": v.skipped} for k, v in asr_stats.items()
            },
        },
        "translation": {
            "to_english_train_file": str((mt_root / "nllb_to_english_train.jsonl").resolve()),
            "to_english_eval_file": str((mt_root / "nllb_to_english_eval.jsonl").resolve()),
            "from_english_train_file": str((mt_root / "nllb_from_english_train.jsonl").resolve()),
            "from_english_eval_file": str((mt_root / "nllb_from_english_eval.jsonl").resolve()),
            "to_english_train_rows": len(mt_to_en_train),
            "to_english_eval_rows": len(mt_to_en_eval),
            "from_english_train_rows": len(mt_from_en_train),
            "from_english_eval_rows": len(mt_from_en_eval),
            "pair_stats": mt_stats,
            "pair_files": {
                pair_name: {
                    "to_english_train_file": str((mt_root / f"{pair_name.replace('-', '_')}_to_english_train.jsonl").resolve()),
                    "to_english_eval_file": str((mt_root / f"{pair_name.replace('-', '_')}_to_english_eval.jsonl").resolve()),
                    "from_english_train_file": str((mt_root / f"english_to_{pair_name.replace('-', '_')}_train.jsonl").resolve()),
                    "from_english_eval_file": str((mt_root / f"english_to_{pair_name.replace('-', '_')}_eval.jsonl").resolve()),
                }
                for pair_name in mt_pair_files.keys()
            },
        },
    }

    summary_path = manifests_root / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
