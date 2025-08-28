import sys, pathlib
# OPTIONAL: only needed if you run this file directly instead of python -m examples.train_classification_model
root = pathlib.Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

import torch
import random
import numpy as np
from src.utils.config import get_small_classifier_config
from src.training.classifier_trainer import SimpleTextDataset, train_classifier, evaluate

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

cfg = get_small_classifier_config()
cfg.num_classes = 2  # binary

# Adjust hyper parameters
cfg.learning_rate = 1e-4
cfg.weight_decay = 0.01
cfg.max_epochs = 8

# Fake tokenized samples (replace with real tokenizer output)
samples = []
for i in range(200):
    length = 10 + (i % 5)
    input_ids = torch.randint(10, 200, (length,), dtype=torch.long)
    attention_mask = torch.ones(length, dtype=torch.long)
    # Easier pattern: label = token > 150 exists
    label = int((input_ids > 150).any())
    samples.append({"input_ids": input_ids, "attention_mask": attention_mask, "label": label})

split = int(0.8 * len(samples))
train_ds = SimpleTextDataset(samples[:split])
val_ds = SimpleTextDataset(samples[split:])

model = train_classifier(cfg, train_ds, val_ds)
torch.save(model.state_dict(), "classifier_model.pt")