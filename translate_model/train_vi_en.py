import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from datasets import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)

DEFAULT_MODEL = "Helsinki-NLP/opus-mt-vi-en"


@dataclass
class TranslationExample:
    source: str
    target: str


def _read_jsonl(path: Path) -> List[TranslationExample]:
    items: List[TranslationExample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw:
                continue
            record = json.loads(raw)
            source = str(record.get("source", "")).strip()
            target = str(record.get("target", "")).strip()
            if not source or not target:
                raise ValueError(f"Invalid sample at line {line_number} in {path}: both source and target are required")
            items.append(TranslationExample(source=source, target=target))
    if not items:
        raise ValueError(f"No samples found in {path}")
    return items


def _load_examples(train_file: Path, eval_file: Optional[Path], validation_split: float, seed: int):
    train_examples = _read_jsonl(train_file)

    if eval_file is not None:
        eval_examples = _read_jsonl(eval_file)
        return train_examples, eval_examples

    if not 0.0 < validation_split < 1.0:
        raise ValueError("validation_split must be between 0 and 1 when eval_file is not provided")

    rng = random.Random(seed)
    shuffled = list(train_examples)
    rng.shuffle(shuffled)
    split_index = max(1, int(len(shuffled) * (1.0 - validation_split)))
    split_index = min(split_index, len(shuffled) - 1)
    return shuffled[:split_index], shuffled[split_index:]


def _build_dataset(examples: Sequence[TranslationExample]) -> Dataset:
    return Dataset.from_list(
        [{"source": item.source, "target": item.target} for item in examples]
    )


def _tokenize_batch(batch, tokenizer, max_source_length: int, max_target_length: int):
    model_inputs = tokenizer(
        batch["source"],
        max_length=max_source_length,
        truncation=True,
    )

    labels = tokenizer(
        text_target=batch["target"],
        max_length=max_target_length,
        truncation=True,
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a Vietnamese-to-English translation model.")
    parser.add_argument("--train_file", required=True, help="Path to training JSONL file")
    parser.add_argument("--eval_file", default=None, help="Optional path to validation JSONL file")
    parser.add_argument("--output_dir", required=True, help="Directory to save checkpoints")
    parser.add_argument("--base_model", default=DEFAULT_MODEL, help=f"Base translation model (default: {DEFAULT_MODEL})")
    parser.add_argument("--max_source_length", type=int, default=128)
    parser.add_argument("--max_target_length", type=int, default=128)
    parser.add_argument("--validation_split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--fp16", action="store_true", help="Enable mixed precision fp16 training")
    parser.add_argument("--bf16", action="store_true", help="Enable mixed precision bf16 training")
    parser.add_argument("--push_to_hub", action="store_true", help="Push the result to Hugging Face Hub")
    parser.add_argument("--hub_model_id", default=None, help="Model id to push to Hugging Face Hub")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    train_file = Path(args.train_file)
    eval_file = Path(args.eval_file) if args.eval_file else None
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_examples, eval_examples = _load_examples(
        train_file=train_file,
        eval_file=eval_file,
        validation_split=args.validation_split,
        seed=args.seed,
    )

    train_dataset = _build_dataset(train_examples)
    eval_dataset = _build_dataset(eval_examples)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.base_model)

    tokenized_train = train_dataset.map(
        lambda batch: _tokenize_batch(batch, tokenizer, args.max_source_length, args.max_target_length),
        batched=True,
        remove_columns=train_dataset.column_names,
    )
    tokenized_eval = eval_dataset.map(
        lambda batch: _tokenize_batch(batch, tokenizer, args.max_source_length, args.max_target_length),
        batched=True,
        remove_columns=eval_dataset.column_names,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        predict_with_generate=True,
        fp16=args.fp16,
        bf16=args.bf16,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        evaluation_strategy="steps",
        save_strategy="steps",
        save_total_limit=args.save_total_limit,
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    eval_metrics = trainer.evaluate()
    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(eval_metrics, handle, indent=2, ensure_ascii=False)

    print(f"Saved fine-tuned model to: {output_dir}")
    print(f"Evaluation metrics written to: {metrics_path}")


if __name__ == "__main__":
    main()
