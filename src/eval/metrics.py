"""Loss functions and evaluation metrics for multimodal sentiment analysis."""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance."""
    
    def __init__(self, alpha: Optional[List[float]] = None, gamma: float = 2.0, reduction: str = "mean"):
        """
        Initialize Focal Loss.
        
        Args:
            alpha: Weighting factors for each class.
            gamma: Focusing parameter.
            reduction: Reduction method ('mean', 'sum', 'none').
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            inputs: Predicted logits.
            targets: Ground truth labels.
            
        Returns:
            torch.Tensor: Focal loss.
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        
        if self.alpha is not None:
            alpha_t = torch.tensor(self.alpha, device=inputs.device)[targets]
            ce_loss = alpha_t * ce_loss
        
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class ContrastiveLoss(nn.Module):
    """Contrastive loss for multimodal alignment."""
    
    def __init__(self, temperature: float = 0.07, margin: float = 1.0):
        """
        Initialize contrastive loss.
        
        Args:
            temperature: Temperature parameter for softmax.
            margin: Margin for hard negative mining.
        """
        super().__init__()
        self.temperature = temperature
        self.margin = margin
    
    def forward(
        self,
        text_embeddings: torch.Tensor,
        audio_embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            text_embeddings: Text embeddings.
            audio_embeddings: Audio embeddings.
            labels: Ground truth labels.
            
        Returns:
            torch.Tensor: Contrastive loss.
        """
        batch_size = text_embeddings.size(0)
        
        # Normalize embeddings
        text_embeddings = F.normalize(text_embeddings, p=2, dim=1)
        audio_embeddings = F.normalize(audio_embeddings, p=2, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(text_embeddings, audio_embeddings.T) / self.temperature
        
        # Create labels for contrastive learning
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float()
        
        # Remove diagonal elements
        mask = mask - torch.eye(batch_size, device=mask.device)
        
        # Compute positive and negative similarities
        pos_sim = torch.sum(similarity_matrix * mask, dim=1)
        neg_sim = torch.sum(similarity_matrix * (1 - mask), dim=1)
        
        # Contrastive loss
        loss = -torch.log(torch.exp(pos_sim) / (torch.exp(pos_sim) + torch.exp(neg_sim)))
        
        return loss.mean()


class MultimodalLoss(nn.Module):
    """Combined loss for multimodal sentiment analysis."""
    
    def __init__(
        self,
        classification_weight: float = 1.0,
        contrastive_weight: float = 0.1,
        use_focal_loss: bool = False,
        alpha: Optional[List[float]] = None,
        gamma: float = 2.0,
        temperature: float = 0.07,
    ):
        """
        Initialize multimodal loss.
        
        Args:
            classification_weight: Weight for classification loss.
            contrastive_weight: Weight for contrastive loss.
            use_focal_loss: Whether to use focal loss.
            alpha: Alpha parameter for focal loss.
            gamma: Gamma parameter for focal loss.
            temperature: Temperature for contrastive loss.
        """
        super().__init__()
        
        self.classification_weight = classification_weight
        self.contrastive_weight = contrastive_weight
        
        if use_focal_loss:
            self.classification_loss = FocalLoss(alpha=alpha, gamma=gamma)
        else:
            self.classification_loss = nn.CrossEntropyLoss()
        
        self.contrastive_loss = ContrastiveLoss(temperature=temperature)
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            outputs: Model outputs containing logits and embeddings.
            labels: Ground truth labels.
            
        Returns:
            Dict containing individual and total losses.
        """
        logits = outputs["logits"]
        text_embeddings = outputs["text_embeddings"]
        audio_embeddings = outputs["audio_embeddings"]
        
        # Classification loss
        classification_loss = self.classification_loss(logits, labels)
        
        # Contrastive loss
        contrastive_loss = self.contrastive_loss(text_embeddings, audio_embeddings, labels)
        
        # Total loss
        total_loss = (
            self.classification_weight * classification_loss +
            self.contrastive_weight * contrastive_loss
        )
        
        return {
            "total_loss": total_loss,
            "classification_loss": classification_loss,
            "contrastive_loss": contrastive_loss,
        }


class SentimentMetrics:
    """Evaluation metrics for sentiment analysis."""
    
    def __init__(self, class_names: Optional[List[str]] = None):
        """
        Initialize metrics.
        
        Args:
            class_names: Names of the classes.
        """
        self.class_names = class_names or ["positive", "negative", "neutral"]
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.predictions = []
        self.targets = []
        self.probabilities = []
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor, probabilities: Optional[torch.Tensor] = None):
        """
        Update metrics with new predictions.
        
        Args:
            predictions: Predicted labels.
            targets: Ground truth labels.
            probabilities: Prediction probabilities.
        """
        self.predictions.extend(predictions.cpu().numpy())
        self.targets.extend(targets.cpu().numpy())
        
        if probabilities is not None:
            self.probabilities.extend(probabilities.cpu().numpy())
    
    def compute(self) -> Dict[str, float]:
        """
        Compute all metrics.
        
        Returns:
            Dict containing computed metrics.
        """
        predictions = np.array(self.predictions)
        targets = np.array(self.targets)
        
        # Basic metrics
        accuracy = accuracy_score(targets, predictions)
        macro_f1 = f1_score(targets, predictions, average="macro")
        weighted_f1 = f1_score(targets, predictions, average="weighted")
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            targets, predictions, average=None, labels=range(len(self.class_names))
        )
        
        # Confusion matrix
        cm = confusion_matrix(targets, predictions, labels=range(len(self.class_names)))
        
        # Classification report
        report = classification_report(
            targets, predictions, target_names=self.class_names, output_dict=True
        )
        
        metrics = {
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "confusion_matrix": cm,
            "classification_report": report,
        }
        
        # Per-class metrics
        for i, class_name in enumerate(self.class_names):
            metrics[f"{class_name}_precision"] = precision[i]
            metrics[f"{class_name}_recall"] = recall[i]
            metrics[f"{class_name}_f1"] = f1[i]
            metrics[f"{class_name}_support"] = support[i]
        
        # Calibration metrics if probabilities are available
        if self.probabilities:
            metrics.update(self._compute_calibration_metrics())
        
        return metrics
    
    def _compute_calibration_metrics(self) -> Dict[str, float]:
        """Compute calibration metrics."""
        probabilities = np.array(self.probabilities)
        targets = np.array(self.targets)
        
        # Expected Calibration Error (ECE)
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (probabilities > bin_lower) & (probabilities <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = (targets[in_bin] == np.argmax(probabilities[in_bin], axis=1)).mean()
                avg_confidence_in_bin = probabilities[in_bin].max(axis=1).mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return {
            "expected_calibration_error": ece,
        }
    
    def get_confusion_matrix_plot(self) -> np.ndarray:
        """Get confusion matrix for plotting."""
        return confusion_matrix(
            self.targets, self.predictions, labels=range(len(self.class_names))
        )


def compute_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    probabilities: Optional[torch.Tensor] = None,
    class_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Compute evaluation metrics.
    
    Args:
        predictions: Predicted labels.
        targets: Ground truth labels.
        probabilities: Prediction probabilities.
        class_names: Names of the classes.
        
    Returns:
        Dict containing computed metrics.
    """
    metrics = SentimentMetrics(class_names)
    metrics.update(predictions, targets, probabilities)
    return metrics.compute()
