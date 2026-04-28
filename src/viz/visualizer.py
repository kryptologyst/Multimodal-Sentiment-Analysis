"""Visualization utilities for multimodal sentiment analysis."""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import torch
from plotly.subplots import make_subplots


class SentimentVisualizer:
    """Visualization class for sentiment analysis results."""
    
    def __init__(self, class_names: Optional[List[str]] = None, save_dir: str = "assets"):
        """
        Initialize visualizer.
        
        Args:
            class_names: Names of the sentiment classes.
            save_dir: Directory to save visualizations.
        """
        self.class_names = class_names or ["positive", "negative", "neutral"]
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use("seaborn-v0_8")
        sns.set_palette("husl")
    
    def plot_confusion_matrix(
        self,
        confusion_matrix: np.ndarray,
        title: str = "Confusion Matrix",
        save: bool = True,
        filename: str = "confusion_matrix.png",
    ) -> plt.Figure:
        """
        Plot confusion matrix.
        
        Args:
            confusion_matrix: Confusion matrix array.
            title: Plot title.
            save: Whether to save the plot.
            filename: Filename to save.
            
        Returns:
            matplotlib Figure object.
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Create heatmap
        sns.heatmap(
            confusion_matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            ax=ax,
        )
        
        ax.set_title(title, fontsize=16, fontweight="bold")
        ax.set_xlabel("Predicted", fontsize=12)
        ax.set_ylabel("Actual", fontsize=12)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.save_dir / filename, dpi=300, bbox_inches="tight")
        
        return fig
    
    def plot_training_history(
        self,
        train_losses: List[float],
        val_losses: List[float],
        val_f1_scores: List[float],
        title: str = "Training History",
        save: bool = True,
        filename: str = "training_history.png",
    ) -> plt.Figure:
        """
        Plot training history.
        
        Args:
            train_losses: Training losses.
            val_losses: Validation losses.
            val_f1_scores: Validation F1 scores.
            title: Plot title.
            save: Whether to save the plot.
            filename: Filename to save.
            
        Returns:
            matplotlib Figure object.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        epochs = range(1, len(train_losses) + 1)
        
        # Plot losses
        ax1.plot(epochs, train_losses, "b-", label="Training Loss", linewidth=2)
        ax1.plot(epochs, val_losses, "r-", label="Validation Loss", linewidth=2)
        ax1.set_title("Training and Validation Loss", fontsize=14, fontweight="bold")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot F1 scores
        ax2.plot(epochs, val_f1_scores, "g-", label="Validation F1", linewidth=2)
        ax2.set_title("Validation F1 Score", fontsize=14, fontweight="bold")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("F1 Score")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16, fontweight="bold")
        plt.tight_layout()
        
        if save:
            plt.savefig(self.save_dir / filename, dpi=300, bbox_inches="tight")
        
        return fig
    
    def plot_class_distribution(
        self,
        labels: List[int],
        title: str = "Class Distribution",
        save: bool = True,
        filename: str = "class_distribution.png",
    ) -> plt.Figure:
        """
        Plot class distribution.
        
        Args:
            labels: List of class labels.
            title: Plot title.
            save: Whether to save the plot.
            filename: Filename to save.
            
        Returns:
            matplotlib Figure object.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Count classes
        unique, counts = np.unique(labels, return_counts=True)
        class_counts = [counts[i] if i in unique else 0 for i in range(len(self.class_names))]
        
        # Create bar plot
        bars = ax.bar(self.class_names, class_counts, color=["#2E8B57", "#DC143C", "#FFD700"])
        
        # Add value labels on bars
        for bar, count in zip(bars, class_counts):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + max(class_counts) * 0.01,
                f"{count}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )
        
        ax.set_title(title, fontsize=16, fontweight="bold")
        ax.set_xlabel("Sentiment Class", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.grid(True, alpha=0.3, axis="y")
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.save_dir / filename, dpi=300, bbox_inches="tight")
        
        return fig
    
    def plot_prediction_confidence(
        self,
        predictions: List[int],
        probabilities: List[np.ndarray],
        title: str = "Prediction Confidence",
        save: bool = True,
        filename: str = "prediction_confidence.png",
    ) -> plt.Figure:
        """
        Plot prediction confidence distribution.
        
        Args:
            predictions: List of predicted labels.
            probabilities: List of prediction probabilities.
            title: Plot title.
            save: Whether to save the plot.
            filename: Filename to save.
            
        Returns:
            matplotlib Figure object.
        """
        fig, axes = plt.subplots(1, len(self.class_names), figsize=(15, 5))
        
        probabilities = np.array(probabilities)
        
        for i, class_name in enumerate(self.class_names):
            # Get confidence scores for this class
            class_confidences = probabilities[:, i]
            
            # Create histogram
            axes[i].hist(class_confidences, bins=20, alpha=0.7, color=f"C{i}")
            axes[i].set_title(f"{class_name.title()} Confidence", fontweight="bold")
            axes[i].set_xlabel("Confidence Score")
            axes[i].set_ylabel("Frequency")
            axes[i].grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16, fontweight="bold")
        plt.tight_layout()
        
        if save:
            plt.savefig(self.save_dir / filename, dpi=300, bbox_inches="tight")
        
        return fig
    
    def plot_attention_weights(
        self,
        attention_weights: torch.Tensor,
        tokens: List[str],
        title: str = "Attention Weights",
        save: bool = True,
        filename: str = "attention_weights.png",
    ) -> plt.Figure:
        """
        Plot attention weights for text tokens.
        
        Args:
            attention_weights: Attention weights tensor.
            tokens: List of token strings.
            title: Plot title.
            save: Whether to save the plot.
            filename: Filename to save.
            
        Returns:
            matplotlib Figure object.
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Convert to numpy and normalize
        weights = attention_weights.cpu().numpy()
        weights = weights / weights.max()  # Normalize to [0, 1]
        
        # Create bar plot
        bars = ax.bar(range(len(tokens)), weights, color="skyblue", alpha=0.7)
        
        # Color bars based on weight
        for bar, weight in zip(bars, weights):
            bar.set_color(plt.cm.Reds(weight))
        
        ax.set_title(title, fontsize=16, fontweight="bold")
        ax.set_xlabel("Token Position")
        ax.set_ylabel("Attention Weight")
        ax.set_xticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45, ha="right")
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.save_dir / filename, dpi=300, bbox_inches="tight")
        
        return fig
    
    def create_interactive_dashboard(
        self,
        metrics: Dict[str, float],
        confusion_matrix: np.ndarray,
        training_history: Dict[str, List[float]],
    ) -> go.Figure:
        """
        Create interactive dashboard with Plotly.
        
        Args:
            metrics: Evaluation metrics.
            confusion_matrix: Confusion matrix.
            training_history: Training history data.
            
        Returns:
            Plotly Figure object.
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Confusion Matrix", "Training History", "Metrics", "Class Distribution"),
            specs=[[{"type": "heatmap"}, {"type": "scatter"}],
                   [{"type": "bar"}, {"type": "bar"}]],
        )
        
        # Confusion Matrix
        fig.add_trace(
            go.Heatmap(
                z=confusion_matrix,
                x=self.class_names,
                y=self.class_names,
                colorscale="Blues",
                showscale=True,
            ),
            row=1, col=1,
        )
        
        # Training History
        epochs = range(1, len(training_history["train_losses"]) + 1)
        fig.add_trace(
            go.Scatter(x=epochs, y=training_history["train_losses"], name="Train Loss", line=dict(color="blue")),
            row=1, col=2,
        )
        fig.add_trace(
            go.Scatter(x=epochs, y=training_history["val_losses"], name="Val Loss", line=dict(color="red")),
            row=1, col=2,
        )
        
        # Metrics
        metric_names = ["Accuracy", "Macro F1", "Weighted F1"]
        metric_values = [metrics["accuracy"], metrics["macro_f1"], metrics["weighted_f1"]]
        fig.add_trace(
            go.Bar(x=metric_names, y=metric_values, name="Metrics"),
            row=2, col=1,
        )
        
        # Class Distribution (placeholder)
        fig.add_trace(
            go.Bar(x=self.class_names, y=[1, 1, 1], name="Classes"),
            row=2, col=2,
        )
        
        # Update layout
        fig.update_layout(
            title="Multimodal Sentiment Analysis Dashboard",
            height=800,
            showlegend=True,
        )
        
        return fig
    
    def save_all_visualizations(
        self,
        metrics: Dict[str, float],
        confusion_matrix: np.ndarray,
        training_history: Dict[str, List[float]],
        labels: List[int],
        predictions: List[int],
        probabilities: List[np.ndarray],
    ):
        """
        Save all visualizations.
        
        Args:
            metrics: Evaluation metrics.
            confusion_matrix: Confusion matrix.
            training_history: Training history data.
            labels: Ground truth labels.
            predictions: Predicted labels.
            probabilities: Prediction probabilities.
        """
        # Confusion Matrix
        self.plot_confusion_matrix(confusion_matrix)
        
        # Training History
        self.plot_training_history(
            training_history["train_losses"],
            training_history["val_losses"],
            training_history["val_f1_scores"],
        )
        
        # Class Distribution
        self.plot_class_distribution(labels)
        
        # Prediction Confidence
        self.plot_prediction_confidence(predictions, probabilities)
        
        # Interactive Dashboard
        dashboard = self.create_interactive_dashboard(metrics, confusion_matrix, training_history)
        dashboard.write_html(self.save_dir / "dashboard.html")
        
        print(f"All visualizations saved to {self.save_dir}")


def visualize_audio_features(
    audio_path: str,
    save_dir: str = "assets",
    filename: str = "audio_features.png",
) -> plt.Figure:
    """
    Visualize audio features (waveform and spectrogram).
    
    Args:
        audio_path: Path to audio file.
        save_dir: Directory to save visualization.
        filename: Filename to save.
        
    Returns:
        matplotlib Figure object.
    """
    import librosa
    import librosa.display
    
    # Load audio
    y, sr = librosa.load(audio_path, sr=16000)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot waveform
    librosa.display.waveshow(y, sr=sr, ax=ax1)
    ax1.set_title("Audio Waveform", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")
    
    # Plot spectrogram
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    img = librosa.display.specshow(D, y_axis="hz", x_axis="time", sr=sr, ax=ax2)
    ax2.set_title("Spectrogram", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Frequency (Hz)")
    
    # Add colorbar
    plt.colorbar(img, ax=ax2, format="%+2.0f dB")
    
    plt.tight_layout()
    
    # Save
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path / filename, dpi=300, bbox_inches="tight")
    
    return fig
