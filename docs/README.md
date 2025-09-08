## Propensity Modelling with Transformer Architecture

Lightweight, from-scratch Transformer components for two common workflows:
- Autoregressive language modeling and text generation
- Sequence classification for tasks like churn/propensity modelling

The code is small, readable, and easy to adapt to your own tokenizers and datasets.


## Quick start

Prereqs: Python 3.10+ and pip. GPU is optional (auto-detected via PyTorch).

Install dependencies (Windows PowerShell):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Optionally run the helper setup script:

```powershell
python .\setup.py
```


## Examples

Train a tiny language model (uses a small config and auto-creates sample data under `data/` if empty):

```powershell
python .\examples\train_simple_model.py
```

Character-level training (no external tokenizer; good for quick demos):

```powershell
python .\examples\train_char_model.py
```

Generate text using the best saved checkpoint (after training):

```powershell
python .\examples\generate_text.py
```

Train a minimal text classifier (binary by default, using synthetic tokenized samples inside the script):

```powershell
python .\examples\train_classification_model.py
```

Notebook: open `Propensity Modelling.ipynb` for a guided, exploratory workflow.


## Using the classifier with your own data

You can train directly from a PyTorch `Dataset`, a pandas `DataFrame`, or a simple list of dict samples. Each sample is:

- input_ids: LongTensor [T]
- attention_mask: LongTensor [T] (1 for real tokens, 0 for padding)
- label: int (class index; for binary you can keep `num_classes=2`)

Minimal pattern:

```python
import torch
from src.utils.config import get_small_classifier_config
from src.training.classifier_trainer import train_classifier

# Build your own tokenized samples
samples = [
		{"input_ids": torch.tensor([12, 34, 56]),
		 "attention_mask": torch.tensor([1, 1, 1]),
		 "label": 1},
		# ... more rows
]

cfg = get_small_classifier_config()
cfg.vocab_size = 50_000  # set to your tokenizer vocab size
cfg.num_classes = 2      # or >2 for multi-class

split = int(0.8 * len(samples))
model = train_classifier(cfg, samples[:split], samples[split:])
```

For convenience with DataFrames, you can pass columns `input_ids`, `label`, and optional `attention_mask`. See `train_classifier_from_dataframe` and `evaluate_from_dataframe` in `src/training/classifier_trainer.py`.


## Architecture overview

- `src/model/transformer.py` — `SimpleLLM` encoder-decoder stack for autoregressive LM with:
	- Token + positional embeddings (`src/model/embeddings.py`)
	- Causal self-attention (`src/model/attention.py`)
	- Pre-norm residual blocks and FFN
	- `.generate(...)` for greedy/sampling/top-k/top-p decoding
- `src/model/classifier.py` — `TransformerClassifier` wraps `SimpleLLM` for sequence classification with CLS/mean pooling.
- `src/generation/generator.py` — `TextGenerator` convenience wrapper for prompting and strategy comparisons.
- `src/training/trainer.py` — `LMTrainer` for language-model training loops, checkpointing, and plotting.
- `src/training/classifier_trainer.py` — utilities: `train_classifier`, early stopping, cross-fold validation, and evaluation.
- `src/training/data_loader.py` — simple text dataset + collate, sample data creation, and split helpers.
- `src/utils/tokenizer.py` — GPT-2 tokenizer wrapper (`transformers`) for wordpiece-like tokens.
- `src/utils/char_tokenizer.py` — dependency-free character tokenizer for quick tests.

Key configs live in `src/utils/config.py` (see `get_small_config()` and `get_small_classifier_config()` to start).


## Project structure

- `examples/` — runnable scripts for training/generation
- `src/model/` — transformer, attention, embeddings, classifier head
- `src/training/` — trainers, loaders, evaluation, CV
- `src/generation/` — generation helpers (prompt, strategies, interactive)
- `src/utils/` — tokenizers and configuration
- `docs/` — concept notes: attention, embeddings, transformer

Explore:
- `docs/attention.md`
- `docs/embeddings.md`
- `docs/transformer.md`


## Propensity modelling tips

- Model customer sequences as token streams (events, attributes, buckets). You can:
	- Use a subword tokenizer (`SimpleTokenizer`) over textual events, or
	- Build a bespoke vocabulary (IDs per event/type) and feed integer tokens directly.
- Provide `attention_mask` to ignore padding for variable-length sessions.
- Start with `get_small_classifier_config()`; scale up dimensions/heads/layers once the pipeline works.

The repo includes `fake_data_gen.py` to produce simple weekly session CSVs you can adapt. Example:

```powershell
python .\fake_data_gen.py --users 50 --weeks 6 --out fake-6-weeks.csv
```


## Notes

- Works on CPU and CUDA; Windows-safe dataloaders (`num_workers=0`).
- Checkpoints are written to `models/`, logs and plots to `logs/`.
- If using `SimpleTokenizer` (GPT-2), the first run downloads tokenizer files via `transformers`.


## License

Please see the repository’s license if provided. If missing, treat as “all rights reserved” until clarified.