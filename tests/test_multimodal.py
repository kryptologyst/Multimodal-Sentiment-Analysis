"""Test suite for multimodal sentiment analysis."""

import pytest
import torch
import numpy as np
from pathlib import Path

from src.models.multimodal_model import MultimodalSentimentModel, TextEncoder, AudioEncoder
from src.eval.metrics import MultimodalLoss, SentimentMetrics
from src.utils.device import get_device, set_seed


class TestMultimodalModel:
    """Test cases for the multimodal model."""
    
    def test_model_initialization(self):
        """Test model initialization."""
        model = MultimodalSentimentModel()
        assert isinstance(model, MultimodalSentimentModel)
        
        # Check model components
        assert hasattr(model, 'text_encoder')
        assert hasattr(model, 'audio_encoder')
        assert hasattr(model, 'fusion')
        assert hasattr(model, 'classifier')
    
    def test_model_forward(self):
        """Test model forward pass."""
        model = MultimodalSentimentModel()
        model.eval()
        
        batch_size = 2
        text_input_ids = torch.randint(0, 1000, (batch_size, 512))
        text_attention_mask = torch.ones(batch_size, 512)
        audio = torch.randn(batch_size, 80, 625)  # mel spectrogram
        
        with torch.no_grad():
            outputs = model(text_input_ids, text_attention_mask, audio)
        
        assert "logits" in outputs
        assert "text_embeddings" in outputs
        assert "audio_embeddings" in outputs
        assert "fused_embeddings" in outputs
        
        assert outputs["logits"].shape == (batch_size, 3)  # 3 classes
        assert outputs["text_embeddings"].shape == (batch_size, 256)
        assert outputs["audio_embeddings"].shape == (batch_size, 256)
    
    def test_text_encoder(self):
        """Test text encoder."""
        encoder = TextEncoder()
        
        batch_size = 2
        input_ids = torch.randint(0, 1000, (batch_size, 512))
        attention_mask = torch.ones(batch_size, 512)
        
        with torch.no_grad():
            embeddings = encoder(input_ids, attention_mask)
        
        assert embeddings.shape == (batch_size, 256)
    
    def test_audio_encoder(self):
        """Test audio encoder."""
        encoder = AudioEncoder()
        
        batch_size = 2
        audio = torch.randn(batch_size, 80, 625)  # mel spectrogram
        
        with torch.no_grad():
            embeddings = encoder(audio)
        
        assert embeddings.shape == (batch_size, 256)


class TestLossFunctions:
    """Test cases for loss functions."""
    
    def test_multimodal_loss(self):
        """Test multimodal loss function."""
        loss_fn = MultimodalLoss()
        
        batch_size = 4
        outputs = {
            "logits": torch.randn(batch_size, 3),
            "text_embeddings": torch.randn(batch_size, 256),
            "audio_embeddings": torch.randn(batch_size, 256),
        }
        labels = torch.randint(0, 3, (batch_size,))
        
        loss_dict = loss_fn(outputs, labels)
        
        assert "total_loss" in loss_dict
        assert "classification_loss" in loss_dict
        assert "contrastive_loss" in loss_dict
        
        assert loss_dict["total_loss"].item() > 0
    
    def test_focal_loss(self):
        """Test focal loss."""
        from src.eval.metrics import FocalLoss
        
        loss_fn = FocalLoss(alpha=[1.0, 1.0, 1.0], gamma=2.0)
        
        batch_size = 4
        inputs = torch.randn(batch_size, 3)
        targets = torch.randint(0, 3, (batch_size,))
        
        loss = loss_fn(inputs, targets)
        assert loss.item() > 0


class TestMetrics:
    """Test cases for evaluation metrics."""
    
    def test_sentiment_metrics(self):
        """Test sentiment metrics computation."""
        metrics = SentimentMetrics()
        
        # Add some test data
        predictions = torch.tensor([0, 1, 2, 0, 1])
        targets = torch.tensor([0, 1, 1, 0, 2])
        probabilities = torch.randn(5, 3)
        
        metrics.update(predictions, targets, probabilities)
        results = metrics.compute()
        
        assert "accuracy" in results
        assert "macro_f1" in results
        assert "weighted_f1" in results
        assert "confusion_matrix" in results
        
        assert 0 <= results["accuracy"] <= 1
        assert 0 <= results["macro_f1"] <= 1
    
    def test_metrics_reset(self):
        """Test metrics reset functionality."""
        metrics = SentimentMetrics()
        
        # Add data
        predictions = torch.tensor([0, 1, 2])
        targets = torch.tensor([0, 1, 1])
        metrics.update(predictions, targets)
        
        # Reset
        metrics.reset()
        
        # Should be empty
        assert len(metrics.predictions) == 0
        assert len(metrics.targets) == 0


class TestDeviceUtils:
    """Test cases for device utilities."""
    
    def test_get_device(self):
        """Test device detection."""
        device = get_device()
        assert isinstance(device, torch.device)
    
    def test_set_seed(self):
        """Test seed setting."""
        set_seed(42)
        
        # Generate some random numbers
        torch_rand = torch.rand(1).item()
        np_rand = np.random.rand()
        
        # Set seed again and generate same numbers
        set_seed(42)
        torch_rand2 = torch.rand(1).item()
        np_rand2 = np.random.rand()
        
        # Should be the same (within floating point precision)
        assert abs(torch_rand - torch_rand2) < 1e-6
        assert abs(np_rand - np_rand2) < 1e-6


class TestDataPipeline:
    """Test cases for data pipeline."""
    
    def test_dataset_creation(self):
        """Test dataset creation."""
        from src.data.dataset import MultimodalSentimentDataset
        
        # Create dataset with synthetic data
        dataset = MultimodalSentimentDataset(
            data_path="dummy_path",  # Will create synthetic data
            split="train"
        )
        
        assert len(dataset) > 0
        
        # Test getting a sample
        sample = dataset[0]
        
        assert "text" in sample
        assert "text_input_ids" in sample
        assert "text_attention_mask" in sample
        assert "audio" in sample
        assert "label" in sample
        
        assert sample["text_input_ids"].shape == (512,)
        assert sample["audio"].shape == (80, 625)
        assert isinstance(sample["label"], torch.Tensor)


if __name__ == "__main__":
    pytest.main([__file__])
