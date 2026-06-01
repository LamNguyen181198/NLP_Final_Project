import argparse
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

DEFAULT_MODEL = "Helsinki-NLP/opus-mt-vi-en"


class VietnameseToEnglishTranslator:
    def __init__(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def translate(self, text: str, max_new_tokens: int = 128) -> str:
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
        ).to(self.device)

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=4,
        )
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Vietnamese-to-English translation inference.")
    parser.add_argument("--model_path", default=DEFAULT_MODEL, help="Path to fine-tuned model or base model id")
    parser.add_argument("--text", default=None, help="Text to translate")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    translator = VietnameseToEnglishTranslator(args.model_path)

    if args.text:
        print(translator.translate(args.text, max_new_tokens=args.max_new_tokens))
        return

    print("Enter Vietnamese text, one line at a time. Press Ctrl+Z then Enter to exit on Windows.")
    while True:
        try:
            line = input("> ").strip()
        except EOFError:
            break
        if not line:
            continue
        print(translator.translate(line, max_new_tokens=args.max_new_tokens))


if __name__ == "__main__":
    main()
