"""Training and evaluation utilities for multimodal sentiment analysis."""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import MultimodalSentimentDataset
from src.eval.metrics import SentimentMetrics, compute_metrics
from src.models.multimodal_model import MultimodalSentimentModel
from src.utils.device import get_device, set_seed


class Trainer:
    """Trainer class for multimodal sentiment analysis."""
    
    def __init__(
        self,
        model: MultimodalSentimentModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: torch.device = None,
        save_dir: str = "checkpoints",
        log_interval: int = 100,
        use_amp: bool = True,
    ):
        """
        Initialize trainer.
        
        Args:
            model: Model to train.
            train_loader: Training data loader.
            val_loader: Validation data loader.
            criterion: Loss function.
            optimizer: Optimizer.
            scheduler: Learning rate scheduler.
            device: Device to use.
            save_dir: Directory to save checkpoints.
            log_interval: Logging interval.
            use_amp: Whether to use automatic mixed precision.
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device or get_device()
        self.save_dir = Path(save_dir)
        self.log_interval = log_interval
        self.use_amp = use_amp
        
        # Create save directory
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Move model to device
        self.model.to(self.device)
        
        # Initialize scaler for mixed precision
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        
        # Initialize metrics
        self.class_names = ["positive", "negative", "neutral"]
        
        # Training state
        self.epoch = 0
        self.best_val_loss = float("inf")
        self.best_val_f1 = 0.0
        self.train_losses = []
        self.val_losses = []
        self.val_f1_scores = []
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        train_metrics = SentimentMetrics(self.class_names)
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            text_input_ids = batch["text_input_ids"].to(self.device)
            text_attention_mask = batch["text_attention_mask"].to(self.device)
            audio = batch["audio"].to(self.device)
            labels = batch["label"].to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(text_input_ids, text_attention_mask, audio)
                    loss_dict = self.criterion(outputs, labels)
                    loss = loss_dict["total_loss"]
                
                # Backward pass
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(text_input_ids, text_attention_mask, audio)
                loss_dict = self.criterion(outputs, labels)
                loss = loss_dict["total_loss"]
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
            
            # Update scheduler
            if self.scheduler:
                self.scheduler.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Get predictions
            predictions = torch.argmax(outputs["logits"], dim=1)
            probabilities = torch.softmax(outputs["logits"], dim=1)
            train_metrics.update(predictions, labels, probabilities)
            
            # Update progress bar
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "avg_loss": f"{total_loss / num_batches:.4f}",
            })
            
            # Log interval
            if batch_idx % self.log_interval == 0:
                logging.info(
                    f"Epoch {self.epoch}, Batch {batch_idx}, "
                    f"Loss: {loss.item():.4f}, "
                    f"Avg Loss: {total_loss / num_batches:.4f}"
                )
        
        # Compute epoch metrics
        epoch_metrics = train_metrics.compute()
        avg_loss = total_loss / num_batches
        
        return {
            "loss": avg_loss,
            "accuracy": epoch_metrics["accuracy"],
            "macro_f1": epoch_metrics["macro_f1"],
            "weighted_f1": epoch_metrics["weighted_f1"],
        }
    
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        val_metrics = SentimentMetrics(self.class_names)
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validation")
            
            for batch in pbar:
                # Move batch to device
                text_input_ids = batch["text_input_ids"].to(self.device)
                text_attention_mask = batch["text_attention_mask"].to(self.device)
                audio = batch["audio"].to(self.device)
                labels = batch["label"].to(self.device)
                
                # Forward pass
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(text_input_ids, text_attention_mask, audio)
                        loss_dict = self.criterion(outputs, labels)
                        loss = loss_dict["total_loss"]
                else:
                    outputs = self.model(text_input_ids, text_attention_mask, audio)
                    loss_dict = self.criterion(outputs, labels)
                    loss = loss_dict["total_loss"]
                
                # Update metrics
                total_loss += loss.item()
                num_batches += 1
                
                # Get predictions
                predictions = torch.argmax(outputs["logits"], dim=1)
                probabilities = torch.softmax(outputs["logits"], dim=1)
                val_metrics.update(predictions, labels, probabilities)
                
                # Update progress bar
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        # Compute validation metrics
        val_metrics_dict = val_metrics.compute()
        avg_loss = total_loss / num_batches
        
        return {
            "loss": avg_loss,
            "accuracy": val_metrics_dict["accuracy"],
            "macro_f1": val_metrics_dict["macro_f1"],
            "weighted_f1": val_metrics_dict["weighted_f1"],
            "confusion_matrix": val_metrics_dict["confusion_matrix"],
            "classification_report": val_metrics_dict["classification_report"],
        }
    
    def train(self, num_epochs: int, patience: int = 3, min_delta: float = 0.001) -> Dict[str, List[float]]:
        """
        Train the model.
        
        Args:
            num_epochs: Number of epochs to train.
            patience: Early stopping patience.
            min_delta: Minimum change to qualify as improvement.
            
        Returns:
            Dict containing training history.
        """
        logging.info(f"Starting training for {num_epochs} epochs")
        
        patience_counter = 0
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            self.train_losses.append(train_metrics["loss"])
            
            # Validate
            val_metrics = self.validate()
            self.val_losses.append(val_metrics["loss"])
            self.val_f1_scores.append(val_metrics["macro_f1"])
            
            # Log epoch results
            logging.info(
                f"Epoch {epoch}: "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Train F1: {train_metrics['macro_f1']:.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Val F1: {val_metrics['macro_f1']:.4f}"
            )
            
            # Save best model
            if val_metrics["loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["loss"]
                self.save_checkpoint("best_loss.pt")
                patience_counter = 0
            elif val_metrics["macro_f1"] > self.best_val_f1:
                self.best_val_f1 = val_metrics["macro_f1"]
                self.save_checkpoint("best_f1.pt")
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                logging.info(f"Early stopping at epoch {epoch}")
                break
        
        # Save final model
        self.save_checkpoint("final.pt")
        
        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "val_f1_scores": self.val_f1_scores,
        }
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": self.epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "best_val_f1": self.best_val_f1,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "val_f1_scores": self.val_f1_scores,
        }
        
        if self.scheduler:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        
        torch.save(checkpoint, self.save_dir / filename)
        logging.info(f"Checkpoint saved: {self.save_dir / filename}")
    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        checkpoint = torch.load(self.save_dir / filename, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if self.scheduler and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        self.epoch = checkpoint["epoch"]
        self.best_val_loss = checkpoint["best_val_loss"]
        self.best_val_f1 = checkpoint["best_val_f1"]
        self.train_losses = checkpoint["train_losses"]
        self.val_losses = checkpoint["val_losses"]
        self.val_f1_scores = checkpoint["val_f1_scores"]
        
        logging.info(f"Checkpoint loaded: {self.save_dir / filename}")


class Evaluator:
    """Evaluator class for multimodal sentiment analysis."""
    
    def __init__(
        self,
        model: MultimodalSentimentModel,
        test_loader: DataLoader,
        device: torch.device = None,
        class_names: Optional[List[str]] = None,
    ):
        """
        Initialize evaluator.
        
        Args:
            model: Model to evaluate.
            test_loader: Test data loader.
            device: Device to use.
            class_names: Names of the classes.
        """
        self.model = model
        self.test_loader = test_loader
        self.device = device or get_device()
        self.class_names = class_names or ["positive", "negative", "neutral"]
        
        # Move model to device
        self.model.to(self.device)
        self.model.eval()
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate the model on test set."""
        logging.info("Starting evaluation")
        
        test_metrics = SentimentMetrics(self.class_names)
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            pbar = tqdm(self.test_loader, desc="Evaluation")
            
            for batch in pbar:
                # Move batch to device
                text_input_ids = batch["text_input_ids"].to(self.device)
                text_attention_mask = batch["text_attention_mask"].to(self.device)
                audio = batch["audio"].to(self.device)
                labels = batch["label"].to(self.device)
                
                # Forward pass
                outputs = self.model(text_input_ids, text_attention_mask, audio)
                
                # Get predictions
                predictions = torch.argmax(outputs["logits"], dim=1)
                probabilities = torch.softmax(outputs["logits"], dim=1)
                
                # Update metrics
                test_metrics.update(predictions, labels, probabilities)
                
                # Store for detailed analysis
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Compute metrics
        metrics = test_metrics.compute()
        
        # Log results
        logging.info(f"Test Accuracy: {metrics['accuracy']:.4f}")
        logging.info(f"Test Macro F1: {metrics['macro_f1']:.4f}")
        logging.info(f"Test Weighted F1: {metrics['weighted_f1']:.4f}")
        
        return metrics
    
    def predict_single(
        self,
        text: str,
        audio_path: Optional[str] = None,
        tokenizer=None,
    ) -> Dict[str, float]:
        """
        Predict sentiment for a single sample.
        
        Args:
            text: Input text.
            audio_path: Path to audio file.
            tokenizer: Text tokenizer.
            
        Returns:
            Dict containing prediction probabilities.
        """
        # Preprocess text
        text_encoding = tokenizer(
            text,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        text_input_ids = text_encoding["input_ids"].to(self.device)
        text_attention_mask = text_encoding["attention_mask"].to(self.device)
        
        # Preprocess audio (simplified for demo)
        if audio_path and os.path.exists(audio_path):
            import librosa
            y, sr = librosa.load(audio_path, sr=16000)
            y = y[:int(10 * sr)]  # Truncate to 10 seconds
            if len(y) < int(10 * sr):
                y = np.pad(y, (0, int(10 * sr) - len(y)), mode="constant")
            
            mel_spec = librosa.feature.melspectrogram(
                y=y, sr=sr, n_mels=80, n_fft=1024, hop_length=256
            )
            log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
            audio = torch.from_numpy(log_mel_spec).float().unsqueeze(0).to(self.device)
        else:
            # Generate dummy audio
            audio = torch.randn(1, 80, 625).to(self.device)  # Dummy mel spectrogram
        
        # Predict
        with torch.no_grad():
            outputs = self.model(text_input_ids, text_attention_mask, audio)
            probabilities = torch.softmax(outputs["logits"], dim=1)
            prediction = torch.argmax(probabilities, dim=1)
        
        # Convert to probabilities
        probs = probabilities.cpu().numpy()[0]
        
        return {
            "prediction": self.class_names[prediction.item()],
            "probabilities": {
                class_name: float(prob)
                for class_name, prob in zip(self.class_names, probs)
            },
        }
