# translate_model

This folder now starts with a multilingual NLLB-200 translation pipeline instead of a single Vietnamese-to-English path.

The practical approach is still the same: start from a pretrained multilingual model, then fine-tune only the language pairs you care about.

## Recommended base model

Default base model:

- `facebook/nllb-200-distilled-600M`

This is a strong starting point for English, German, Japanese, Vietnamese, and many other languages.

## Files

- `train_nllb.py` - fine-tune NLLB on any supported language pair
- `infer_nllb.py` - run inference from a saved checkpoint or the base model
- `nllb.py` - shared multilingual model helper and language code mapping
- `data/README.md` - expected dataset format

## Language codes

Use NLLB language codes for source and target languages, for example:

- English: `eng_Latn`
- German: `deu_Latn`
- Japanese: `jpn_Jpan`
- Vietnamese: `vie_Latn`

## Inference

Translate German to English:

```bash
python infer_nllb.py \
  --source_lang deu_Latn \
  --target_lang eng_Latn \
  --text "Guten Morgen, wie geht es dir?"
```

Translate English to Japanese:

```bash
python infer_nllb.py \
  --source_lang eng_Latn \
  --target_lang jpn_Jpan \
  --text "Good morning, how are you?"
```

## Training

Use JSONL with one sample per line:

```jsonl
{"source": "Guten Morgen.", "target": "Good morning."}
{"source": "今日はいい天気です。", "target": "The weather is nice today."}
```

Fine-tune German to English:

```bash
python train_nllb.py \
  --train_file ./data/train.jsonl \
  --eval_file ./data/valid.jsonl \
  --output_dir ./outputs/de-en-translator \
  --source_lang deu_Latn \
  --target_lang eng_Latn
```

If you do not have a validation file, the script can split the training data automatically.

## Next step

After the NLLB path works, connect the output of Qwen3-ASR directly into this translator from the frontend API and add a dropdown later if you want manual target-language selection.
