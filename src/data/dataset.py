"""Data loading and preprocessing utilities for multimodal sentiment analysis."""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import librosa
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class MultimodalSentimentDataset(Dataset):
    """
    Dataset class for multimodal sentiment analysis.
    
    Handles text and audio data with proper preprocessing and augmentation.
    """
    
    def __init__(
        self,
        data_path: Union[str, Path],
        tokenizer_name: str = "distilbert-base-uncased",
        max_text_length: int = 512,
        max_audio_length: float = 10.0,
        sample_rate: int = 16000,
        n_mels: int = 80,
        n_fft: int = 1024,
        hop_length: int = 256,
        augment: bool = False,
        split: str = "train",
    ):
        """
        Initialize the dataset.
        
        Args:
            data_path: Path to the dataset directory or JSON file.
            tokenizer_name: Name of the tokenizer to use.
            max_text_length: Maximum text sequence length.
            max_audio_length: Maximum audio length in seconds.
            sample_rate: Audio sample rate.
            n_mels: Number of mel frequency bins.
            n_fft: FFT window size.
            hop_length: Hop length for STFT.
            augment: Whether to apply data augmentation.
            split: Dataset split (train, val, test).
        """
        self.data_path = Path(data_path)
        self.max_text_length = max_text_length
        self.max_audio_length = max_audio_length
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.augment = augment
        self.split = split
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Load data
        self.data = self._load_data()
        
        # Sentiment label mapping
        self.label_map = {"positive": 0, "negative": 1, "neutral": 2}
        self.reverse_label_map = {v: k for k, v in self.label_map.items()}
    
    def _load_data(self) -> List[Dict[str, Any]]:
        """Load data from the specified path."""
        if self.data_path.is_file() and self.data_path.suffix == ".json":
            # Load from JSON file
            with open(self.data_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        elif self.data_path.is_dir():
            # Load from directory structure
            data = self._load_from_directory()
        else:
            # Create synthetic data for demonstration
            data = self._create_synthetic_data()
        
        # Filter by split if specified
        if "split" in data[0]:
            data = [item for item in data if item["split"] == self.split]
        
        return data
    
    def _load_from_directory(self) -> List[Dict[str, Any]]:
        """Load data from directory structure."""
        data = []
        
        # Look for annotations file
        annotations_file = self.data_path / "annotations.json"
        if annotations_file.exists():
            with open(annotations_file, "r", encoding="utf-8") as f:
                annotations = json.load(f)
            
            for item in annotations:
                text_file = self.data_path / "text" / f"{item['id']}.txt"
                audio_file = self.data_path / "audio" / f"{item['id']}.wav"
                
                if text_file.exists() and audio_file.exists():
                    with open(text_file, "r", encoding="utf-8") as f:
                        text = f.read().strip()
                    
                    data.append({
                        "id": item["id"],
                        "text": text,
                        "audio_path": str(audio_file),
                        "label": item["label"],
                        "split": item.get("split", "train"),
                    })
        
        return data
    
    def _create_synthetic_data(self) -> List[Dict[str, Any]]:
        """Create synthetic data for demonstration purposes."""
        synthetic_data = [
            {
                "id": "sample_1",
                "text": "I absolutely love this product! It's amazing and works perfectly.",
                "audio_path": None,  # Will be generated on-the-fly
                "label": "positive",
                "split": "train",
            },
            {
                "id": "sample_2", 
                "text": "This is terrible. I hate it and it doesn't work at all.",
                "audio_path": None,
                "label": "negative",
                "split": "train",
            },
            {
                "id": "sample_3",
                "text": "It's okay, nothing special but it works.",
                "audio_path": None,
                "label": "neutral",
                "split": "val",
            },
            {
                "id": "sample_4",
                "text": "Fantastic! This exceeded all my expectations.",
                "audio_path": None,
                "label": "positive",
                "split": "test",
            },
            {
                "id": "sample_5",
                "text": "I'm disappointed with the quality and service.",
                "audio_path": None,
                "label": "negative",
                "split": "test",
            },
        ]
        
        return synthetic_data
    
    def _preprocess_text(self, text: str) -> Dict[str, torch.Tensor]:
        """Preprocess text input."""
        # Tokenize text
        encoding = self.tokenizer(
            text,
            max_length=self.max_text_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }
    
    def _preprocess_audio(self, audio_path: Optional[str]) -> torch.Tensor:
        """Preprocess audio input."""
        if audio_path and os.path.exists(audio_path):
            # Load audio file
            y, sr = librosa.load(audio_path, sr=self.sample_rate)
        else:
            # Generate synthetic audio based on text sentiment
            # This is a simplified approach for demonstration
            duration = min(self.max_audio_length, 5.0)
            y = np.random.randn(int(duration * self.sample_rate)) * 0.1
        
        # Truncate or pad to max length
        max_samples = int(self.max_audio_length * self.sample_rate)
        if len(y) > max_samples:
            y = y[:max_samples]
        else:
            y = np.pad(y, (0, max_samples - len(y)), mode="constant")
        
        # Extract mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=y,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )
        
        # Convert to log scale
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Convert to tensor
        audio_tensor = torch.from_numpy(log_mel_spec).float()
        
        return audio_tensor
    
    def _augment_audio(self, audio_tensor: torch.Tensor) -> torch.Tensor:
        """Apply audio augmentation."""
        if not self.augment:
            return audio_tensor
        
        # Add noise
        noise_factor = 0.01
        noise = torch.randn_like(audio_tensor) * noise_factor
        audio_tensor = audio_tensor + noise
        
        # Time masking
        if torch.rand(1) < 0.5:
            mask_length = int(audio_tensor.shape[1] * 0.1)
            mask_start = torch.randint(0, audio_tensor.shape[1] - mask_length, (1,))
            audio_tensor[:, mask_start:mask_start + mask_length] = 0
        
        return audio_tensor
    
    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single item from the dataset."""
        item = self.data[idx]
        
        # Preprocess text
        text_data = self._preprocess_text(item["text"])
        
        # Preprocess audio
        audio_tensor = self._preprocess_audio(item["audio_path"])
        
        # Apply augmentation if training
        if self.augment and self.split == "train":
            audio_tensor = self._augment_audio(audio_tensor)
        
        # Get label
        label = self.label_map[item["label"]]
        
        return {
            "id": item["id"],
            "text": item["text"],
            "text_input_ids": text_data["input_ids"],
            "text_attention_mask": text_data["attention_mask"],
            "audio": audio_tensor,
            "label": torch.tensor(label, dtype=torch.long),
        }


def create_data_loaders(
    data_path: Union[str, Path],
    batch_size: int = 32,
    num_workers: int = 4,
    tokenizer_name: str = "distilbert-base-uncased",
    max_text_length: int = 512,
    max_audio_length: float = 10.0,
    sample_rate: int = 16000,
    n_mels: int = 80,
    n_fft: int = 1024,
    hop_length: int = 256,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create data loaders for train, validation, and test splits.
    
    Args:
        data_path: Path to the dataset.
        batch_size: Batch size for data loaders.
        num_workers: Number of worker processes.
        tokenizer_name: Name of the tokenizer.
        max_text_length: Maximum text sequence length.
        max_audio_length: Maximum audio length in seconds.
        sample_rate: Audio sample rate.
        n_mels: Number of mel frequency bins.
        n_fft: FFT window size.
        hop_length: Hop length for STFT.
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    # Create datasets for each split
    train_dataset = MultimodalSentimentDataset(
        data_path=data_path,
        tokenizer_name=tokenizer_name,
        max_text_length=max_text_length,
        max_audio_length=max_audio_length,
        sample_rate=sample_rate,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        augment=True,
        split="train",
    )
    
    val_dataset = MultimodalSentimentDataset(
        data_path=data_path,
        tokenizer_name=tokenizer_name,
        max_text_length=max_text_length,
        max_audio_length=max_audio_length,
        sample_rate=sample_rate,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        augment=False,
        split="val",
    )
    
    test_dataset = MultimodalSentimentDataset(
        data_path=data_path,
        tokenizer_name=tokenizer_name,
        max_text_length=max_text_length,
        max_audio_length=max_audio_length,
        sample_rate=sample_rate,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        augment=False,
        split="test",
    )
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader, test_loader
