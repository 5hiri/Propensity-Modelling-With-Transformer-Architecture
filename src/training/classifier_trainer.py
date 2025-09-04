import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from dataclasses import asdict
import math
import time
from typing import Union, Optional, Any, TYPE_CHECKING

try:  # Optional pandas support
    import pandas as pd  # type: ignore
    _PANDAS_AVAILABLE = True
except ImportError:  # pragma: no cover - pandas optional
    pd = None  # type: ignore
    _PANDAS_AVAILABLE = False

if TYPE_CHECKING:
    import pandas as _pd
    DataFrameLike = _pd.DataFrame
else:  # at runtime keep it loose to avoid hard dependency
    DataFrameLike = Any

from src.utils.config import ModelConfig
from src.model.transformer import SimpleLLM
from src.model.classifier import TransformerClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold, StratifiedKFold
import numpy as np


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

def _dataframe_to_dataset(df: DataFrameLike) -> SimpleTextDataset:
    """Convert a pandas DataFrame into a SimpleTextDataset.

    Expected columns:
      - input_ids: sequence (list[int] or 1D tensor)
      - attention_mask: (optional) sequence same length as input_ids
      - label: class index / int
    Any missing attention_mask column will be auto-created as ones.
    """
    if not _PANDAS_AVAILABLE:
        raise RuntimeError("pandas is not installed but a DataFrame was provided.")
    required = {"input_ids", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing required columns: {missing}")
    has_mask = "attention_mask" in df.columns
    samples = []
    for _, row in df.iterrows():
        input_ids = row["input_ids"]
        if not torch.is_tensor(input_ids):
            input_ids = torch.tensor(list(input_ids), dtype=torch.long)
        attention_mask = row["attention_mask"] if has_mask else torch.ones_like(input_ids)
        if not torch.is_tensor(attention_mask):
            attention_mask = torch.tensor(list(attention_mask), dtype=torch.long)
        label = int(row["label"])  # enforce int
        samples.append({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": label,
        })
    return SimpleTextDataset(samples)


def _ensure_dataset(obj: Union[Dataset, DataFrameLike, list, tuple]):
    """Accept several lightweight container formats and return a Dataset.

    Supported:
      - torch.utils.data.Dataset (returned as-is)
      - pandas.DataFrame (converted)
      - list/tuple of dict samples (wrapped)
    """
    if isinstance(obj, Dataset):
        return obj
    if _PANDAS_AVAILABLE and isinstance(obj, pd.DataFrame):  # type: ignore
        return _dataframe_to_dataset(obj)
    if isinstance(obj, (list, tuple)) and obj and isinstance(obj[0], dict):
        return SimpleTextDataset(obj)
    raise TypeError("Unsupported dataset type. Provide a Dataset, DataFrame, or list of sample dicts.")


def train_classifier(cfg: ModelConfig,
                     train_dataset: Union[Dataset, DataFrameLike, list, tuple],
                     val_dataset: Optional[Union[Dataset, DataFrameLike, list, tuple]] = None,
                     device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    """Train a classifier.

    train_dataset / val_dataset can be:
      - torch Dataset returning (input_ids, attention_mask, label)
      - pandas DataFrame with columns: input_ids, (optional) attention_mask, label
      - list/tuple of dict samples {"input_ids": Tensor|list[int], "attention_mask": Tensor|list[int], "label": int}
    """
    train_dataset = _ensure_dataset(train_dataset)
    val_dataset = _ensure_dataset(val_dataset) if val_dataset is not None else None

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

            if global_step % 1000 == 0:
                preds = out["preds"]
                acc = (preds.view(-1) == labels.view(-1)).float().mean().item()
                lr = scheduler.get_last_lr()[0]
                elapsed = time.time() - start_time
                print(f"epoch {epoch} step {global_step} lr {lr:.2e} loss {loss.item():.4f} acc {acc:.4f} elapsed {elapsed:.1f}s")

            global_step += 1

        if val_loader:
            val_result = evaluate(model, val_loader, device, return_metrics=True)
            val_loss, val_acc, val_cm, val_report = val_result
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


def train_classifier_from_dataframe(cfg: ModelConfig,
                                    train_df: DataFrameLike,
                                    val_df: Optional[DataFrameLike] = None,
                                    device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    """Convenience wrapper when working directly with pandas DataFrames."""
    if not _PANDAS_AVAILABLE:
        raise RuntimeError("pandas not installed. Install pandas or pass a Dataset instead.")
    return train_classifier(cfg, train_df, val_df, device=device)

@torch.no_grad()
def evaluate(model: torch.nn.Module, data_loader: DataLoader, device: str, return_metrics: bool = False):
    model.eval()
    total, correct, total_loss = 0, 0, 0.0
    all_preds = []
    all_labels = []
    
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
        
        # Collect predictions and labels for confusion matrix
        all_preds.extend(preds.view(-1).cpu().numpy())
        all_labels.extend(labels.view(-1).cpu().numpy())
    
    avg_loss = total_loss / total
    acc = correct / total
    
    if not return_metrics:
        print(f"val loss {avg_loss:.4f} acc {acc:.4f}")
        model.train()
        return None
    
    # Import classification_report to get precision, recall, and support
    
    # Create confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Get precision, recall, and support
    report = classification_report(all_labels, all_preds, output_dict=True)
    
    model.train()
    return (avg_loss, acc, cm, report)


@torch.no_grad()
def evaluate_from_dataframe(model: torch.nn.Module, 
                           df: DataFrameLike, 
                           device: str, 
                           batch_size: int = 32,
                           return_metrics: bool = False):
    """Evaluate model using a pandas DataFrame directly."""
    if not _PANDAS_AVAILABLE:
        raise RuntimeError("pandas not installed. Install pandas or use evaluate() with DataLoader.")
    
    dataset = _dataframe_to_dataset(df)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_batch
    )
    
    return evaluate(model, data_loader, device, return_metrics)


@torch.no_grad()
def cross_fold_validation(cfg: ModelConfig,
                         dataset: Union[Dataset, DataFrameLike, list, tuple],
                         n_splits: int = 5,
                         stratified: bool = True,
                         device: str = "cuda" if torch.cuda.is_available() else "cpu",
                         random_state: int = 42):
    """
    Perform k-fold cross validation on the dataset.
    
    Args:
        cfg: Model configuration
        dataset: Dataset for cross validation
        n_splits: Number of folds for cross validation
        stratified: Whether to use stratified k-fold (maintains class distribution)
        device: Device to run training on
        random_state: Random state for reproducibility
        
    Returns:
        Dictionary containing cross validation results
    """
    print(f"Starting {n_splits}-fold cross validation...")
    
    # Ensure dataset is in the right format
    dataset = _ensure_dataset(dataset)
    
    # Extract data for cross validation
    all_samples = []
    all_labels = []
    
    for i in range(len(dataset)):
        input_ids, attention_mask, label = dataset[i]
        all_samples.append({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": label
        })
        all_labels.append(label)
    
    # Setup cross validation
    if stratified:
        kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        splits = list(kfold.split(all_samples, all_labels))
    else:
        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        splits = list(kfold.split(all_samples))
    
    # Store results for each fold
    fold_results = []
    all_accuracies = []
    all_losses = []
    all_f1_scores = []
    all_precisions = []
    all_recalls = []
    
    for fold, (train_idx, val_idx) in enumerate(splits):
        print(f"\n=== Fold {fold + 1}/{n_splits} ===")
        
        # Split data for this fold
        train_samples = [all_samples[i] for i in train_idx]
        val_samples = [all_samples[i] for i in val_idx]
        
        train_dataset_fold = SimpleTextDataset(train_samples)
        val_dataset_fold = SimpleTextDataset(val_samples)
        
        # Train model for this fold
        print(f"Training fold {fold + 1}...")
        
        # Create a copy of the config for this fold to avoid modifying the original
        from dataclasses import replace
        fold_cfg = replace(cfg, max_epochs=max(1, cfg.max_epochs // 2))  # Reduce epochs for CV
        
        model = train_classifier(
            cfg=fold_cfg,
            train_dataset=train_dataset_fold,
            val_dataset=None,  # Don't use validation during CV training
            device=device
        )
        
        # Evaluate on validation set
        print(f"Evaluating fold {fold + 1}...")
        val_loader_fold = DataLoader(
            val_dataset_fold,
            batch_size=cfg.batch_size,
            shuffle=False,
            collate_fn=collate_batch
        )
        
        # Call evaluate with return_metrics=True and handle the return properly
        eval_result = evaluate(model, val_loader_fold, device, return_metrics=True)
        if eval_result is not None:
            val_loss, accuracy, cm, report = eval_result
            
            # Extract metrics safely with proper type checking
            if isinstance(report, dict):
                macro_avg = report.get('macro avg', {})
                weighted_avg = report.get('weighted avg', {})
                
                if isinstance(macro_avg, dict):
                    macro_f1 = macro_avg.get('f1-score', 0.0)
                    precision_macro = macro_avg.get('precision', 0.0)
                    recall_macro = macro_avg.get('recall', 0.0)
                else:
                    macro_f1 = precision_macro = recall_macro = 0.0
                    
                if isinstance(weighted_avg, dict):
                    weighted_f1 = weighted_avg.get('f1-score', 0.0)
                else:
                    weighted_f1 = 0.0
            else:
                # If report is not a dict, use default values
                macro_f1 = weighted_f1 = precision_macro = recall_macro = 0.0
        else:
            # Fallback if evaluation fails
            val_loss, accuracy = 0.0, 0.0
            macro_f1 = weighted_f1 = precision_macro = recall_macro = 0.0
            cm = np.zeros((2, 2))
            report = {}
        
        # Store results
        fold_result = {
            'fold': fold + 1,
            'val_loss': val_loss,
            'accuracy': accuracy,
            'f1_macro': macro_f1,
            'f1_weighted': weighted_f1,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'confusion_matrix': cm,
            'classification_report': report
        }
        fold_results.append(fold_result)
        
        # Collect for averaging
        all_accuracies.append(accuracy)
        all_losses.append(val_loss)
        all_f1_scores.append(macro_f1)
        all_precisions.append(precision_macro)
        all_recalls.append(recall_macro)
        
        # Print fold results
        print(f"Fold {fold + 1} Results:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Loss: {val_loss:.4f}")
        print(f"  F1 (macro): {macro_f1:.4f}")
        print(f"  Precision (macro): {precision_macro:.4f}")
        print(f"  Recall (macro): {recall_macro:.4f}")
    
    # Calculate overall statistics
    mean_accuracy = np.mean(all_accuracies)
    std_accuracy = np.std(all_accuracies)
    mean_loss = np.mean(all_losses)
    std_loss = np.std(all_losses)
    mean_f1 = np.mean(all_f1_scores)
    std_f1 = np.std(all_f1_scores)
    mean_precision = np.mean(all_precisions)
    std_precision = np.std(all_precisions)
    mean_recall = np.mean(all_recalls)
    std_recall = np.std(all_recalls)
    
    # Print summary
    print(f"\n=== Cross Validation Summary ({n_splits}-fold) ===")
    print(f"Accuracy:  {mean_accuracy:.4f} ± {std_accuracy:.4f}")
    print(f"Loss:      {mean_loss:.4f} ± {std_loss:.4f}")
    print(f"F1 Score:  {mean_f1:.4f} ± {std_f1:.4f}")
    print(f"Precision: {mean_precision:.4f} ± {std_precision:.4f}")
    print(f"Recall:    {mean_recall:.4f} ± {std_recall:.4f}")
    
    # Return comprehensive results
    return {
        'fold_results': fold_results,
        'summary': {
            'mean_accuracy': mean_accuracy,
            'std_accuracy': std_accuracy,
            'mean_loss': mean_loss,
            'std_loss': std_loss,
            'mean_f1': mean_f1,
            'std_f1': std_f1,
            'mean_precision': mean_precision,
            'std_precision': std_precision,
            'mean_recall': mean_recall,
            'std_recall': std_recall,
            'n_splits': n_splits,
            'stratified': stratified
        }
    }