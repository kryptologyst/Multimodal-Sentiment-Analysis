"""Multimodal Sentiment Analysis Package."""

__version__ = "1.0.0"
__author__ = "AI Research"
__email__ = "research@example.com"

from .models.multimodal_model import MultimodalSentimentModel
from .eval.metrics import MultimodalLoss, SentimentMetrics
from .data.dataset import MultimodalSentimentDataset, create_data_loaders
from .utils.device import get_device, set_seed

__all__ = [
    "MultimodalSentimentModel",
    "MultimodalLoss", 
    "SentimentMetrics",
    "MultimodalSentimentDataset",
    "create_data_loaders",
    "get_device",
    "set_seed",
]
