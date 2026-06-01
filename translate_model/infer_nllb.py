import argparse

from translate_model.nllb import DEFAULT_MODEL_ID, NLLBTranslator, build_argument_parser


def parse_args() -> argparse.Namespace:
    parser = build_argument_parser("Run NLLB-200 translation inference.")
    parser.set_defaults(model_id=DEFAULT_MODEL_ID)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    translator = NLLBTranslator(
        model_id=args.model_id,
        source_lang=args.source_lang,
        target_lang=args.target_lang,
    )

    if args.text:
        print(
            translator.translate(
                args.text,
                source_lang=args.source_lang,
                target_lang=args.target_lang,
                max_new_tokens=args.max_new_tokens,
            )
        )
        return

    print("Enter text one line at a time. Press Ctrl+Z then Enter to exit on Windows.")
    while True:
        try:
            line = input("> ").strip()
        except EOFError:
            break
        if not line:
            continue
        print(
            translator.translate(
                line,
                source_lang=args.source_lang,
                target_lang=args.target_lang,
                max_new_tokens=args.max_new_tokens,
            )
        )


if __name__ == "__main__":
    main()