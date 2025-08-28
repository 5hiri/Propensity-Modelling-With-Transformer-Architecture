import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from dataclasses import asdict
import math
import time

from src.utils.config import ModelConfig
from src.model.transformer import SimpleLLM
from src.model.classifier import TransformerClassifier


class SimpleTextDataset(Dataset):
    """
    Expects each sample as dict with:
      "input_ids": LongTensor [T]
      "attention_mask": LongTensor [T] (1 for real tokens, 0 for pad)
      "label": int (0/1 or class index)
    """
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return (s["input_ids"], s["attention_mask"], s["label"])


def collate_batch(batch, pad_id: int = 0):
    # batch: list of tuples
    input_ids_list, mask_list, labels_list = zip(*batch)
    max_len = max(x.size(0) for x in input_ids_list)

    def pad(t):
        if t.size(0) == max_len:
            return t
        pad_len = max_len - t.size(0)
        return torch.cat([t, torch.full((pad_len,), pad_id, dtype=torch.long)], dim=0)

    input_ids = torch.stack([pad(x) for x in input_ids_list], dim=0)        # [B, T]
    attention_mask = torch.stack([pad(m) for m in mask_list], dim=0)        # [B, T]
    labels = torch.tensor(labels_list, dtype=torch.long)                    # [B]
    return input_ids, attention_mask, labels


def build_model(cfg: ModelConfig, single_logit: bool = True):
    backbone = SimpleLLM(
        vocab_size=cfg.vocab_size,
        d_model=cfg.d_model,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        d_ff=cfg.d_ff,
        max_seq_len=cfg.max_seq_len,
        dropout=cfg.dropout,
        use_learned_pe=False
    )
    model = TransformerClassifier(
        backbone=backbone,
        num_classes=cfg.num_classes,
        pooling="mean",           # or "cls"
        single_logit=single_logit
    )
    return model


def train_classifier(cfg: ModelConfig,
                     train_dataset: Dataset,
                     val_dataset: Dataset | None = None,
                     device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    model = build_model(cfg, single_logit=(cfg.num_classes == 2))
    model.to(device)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate_batch
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            collate_fn=collate_batch
        )

    optimizer = AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    
    total_steps = cfg.max_epochs * math.ceil(len(train_loader))
    warmup = max(10, int(0.05 * total_steps))

    def lr_lambda(step):
        if step < warmup:
            return step / float(max(1, warmup))
        progress = (step - warmup) / float(max(1, total_steps - warmup))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val_loss = float("inf")
    best_state = None
    global_step = 0
    start_time = time.time()

    for epoch in range(cfg.max_epochs):
        model.train()
        for batch_idx, (input_ids, attention_mask, labels) in enumerate(train_loader):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            out = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = out["loss"]

            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            if global_step % 100 == 0:
                preds = out["preds"]
                acc = (preds.view(-1) == labels.view(-1)).float().mean().item()
                lr = scheduler.get_last_lr()[0]
                elapsed = time.time() - start_time
                print(f"epoch {epoch} step {global_step} lr {lr:.2e} loss {loss.item():.4f} acc {acc:.4f} elapsed {elapsed:.1f}s")

            global_step += 1

        # End-of-epoch validation
        if val_loader:
            val_loss, val_acc = evaluate(model, val_loader, device, return_metrics=True)
            improved = val_loss < best_val_loss - 1e-4
            if improved:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                print(f"[best] val_loss {val_loss:.4f} acc {val_acc:.4f}")
            else:
                print(f"val_loss {val_loss:.4f} acc {val_acc:.4f}")

    if best_state:
        model.load_state_dict(best_state)

    return model

@torch.no_grad()
def evaluate(model: torch.nn.Module, data_loader: DataLoader, device: str, return_metrics: bool = False):
    model.eval()
    total, correct, total_loss = 0, 0, 0.0
    for input_ids, attention_mask, labels in data_loader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)
        out = model(input_ids, attention_mask=attention_mask, labels=labels)
        bsz = labels.size(0)
        total_loss += out["loss"].item() * bsz
        preds = out["preds"]
        correct += (preds.view(-1) == labels.view(-1)).sum().item()
        total += bsz
    avg_loss = total_loss / total
    acc = correct / total
    if not return_metrics:
        print(f"val loss {avg_loss:.4f} acc {acc:.4f}")
    model.train()
    return (avg_loss, acc) if return_metrics else None