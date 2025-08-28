import torch
import torch.nn as nn
import torch.nn.functional as F
from .transformer import SimpleLLM

class TransformerClassifier(nn.Module):
    """
    Classification wrapper around SimpleLLM encoder.
    
    Args:
        backbone: SimpleLLM instance.
        num_classes: Number of target classes.
        pooling: 'cls' (use first token) or 'mean' (masked mean over tokens).
        single_logit: If True and num_classes == 2, produce a single logit
                      (use BCEWithLogits); otherwise produce num_classes logits
                      (use CrossEntropy).
    """
    def __init__(self,
                 backbone: SimpleLLM,
                 num_classes: int = 2,
                 pooling: str = "cls",
                 single_logit: bool = True):
        super().__init__()
        assert pooling in ("cls", "mean"), "pooling must be 'cls' or 'mean'"
        if single_logit and num_classes != 2:
            raise ValueError("single_logit=True is only valid for binary classification (num_classes=2)")
        self.backbone = backbone
        self.pooling = pooling
        self.num_classes = num_classes
        out_dim = 1 if (single_logit and num_classes == 2) else num_classes
        self.head = nn.Linear(backbone.d_model, out_dim)
        self.single_logit = (out_dim == 1)

    def _pool(self, hidden: torch.Tensor, attention_mask: torch.Tensor | None):
        # hidden: [B, T, D]
        if self.pooling == "cls":
            # Assumes first token acts as CLS (caller must ensure this)
            return hidden[:, 0]
        else:
            # Masked mean pooling (ignore padding if attention_mask provided)
            if attention_mask is None:
                return hidden.mean(dim=1)
            mask = attention_mask.float()  # [B, T]
            mask = mask.unsqueeze(-1)      # [B, T, 1]
            summed = (hidden * mask).sum(dim=1)  # [B, D]
            denom = mask.sum(dim=1).clamp(min=1e-6)  # [B, 1]
            return summed / denom

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor | None = None,
                labels: torch.Tensor | None = None):
        # Get encoder hidden states (skip vocab projection)
        outputs = self.backbone(
            input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            return_hidden_states=True,
            compute_logits=False
        )
        hidden_states = outputs["hidden_states"]  # [B, T, D]

        pooled = self._pool(hidden_states, attention_mask)  # [B, D]
        logits = self.head(pooled)  # [B, 1] or [B, C]

        result = {"logits": logits}

        if labels is None:
            if self.single_logit:
                result["probs"] = torch.sigmoid(logits)
            else:
                result["probs"] = F.softmax(logits, dim=-1)
            return result

        # Compute loss
        if self.single_logit:
            # labels expected shape [B] (0/1) or [B, 1]
            labels_float = labels.float().view(-1)
            loss = F.binary_cross_entropy_with_logits(logits.view(-1), labels_float)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).long()
        else:
            # labels expected shape [B]
            loss = F.cross_entropy(logits, labels.long())
            probs = F.softmax(logits, dim=-1)
            preds = probs.argmax(dim=-1)

        result.update({
            "loss": loss,
            "probs": probs,
            "preds": preds
        })
        return result