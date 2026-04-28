"""Evaluation script for multimodal sentiment analysis."""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Any

import torch
from torch.utils.data import DataLoader

from src.data.dataset import MultimodalSentimentDataset
from src.eval.trainer import Evaluator
from src.models.multimodal_model import MultimodalSentimentModel
from src.utils.device import get_device, set_seed
from src.viz.visualizer import SentimentVisualizer


def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def load_model(model_path: str, device: torch.device) -> MultimodalSentimentModel:
    """Load trained model from checkpoint."""
    checkpoint = torch.load(model_path, map_location=device)
    
    # Create model
    model = MultimodalSentimentModel()
    
    # Load state dict
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    return model


def create_test_loader(data_path: str, batch_size: int = 32) -> DataLoader:
    """Create test data loader."""
    dataset = MultimodalSentimentDataset(
        data_path=data_path,
        split="test",
        augment=False,
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    
    return loader


def evaluate_model(
    model_path: str,
    data_path: str,
    output_dir: str,
    batch_size: int = 32,
    device: str = "auto",
) -> Dict[str, Any]:
    """Evaluate model on test set."""
    # Setup
    device = get_device(device)
    set_seed(42, deterministic=True)
    
    logging.info(f"Using device: {device}")
    logging.info(f"Model path: {model_path}")
    logging.info(f"Data path: {data_path}")
    
    # Load model
    model = load_model(model_path, device)
    logging.info("Model loaded successfully")
    
    # Create test loader
    test_loader = create_test_loader(data_path, batch_size)
    logging.info(f"Test samples: {len(test_loader.dataset)}")
    
    # Create evaluator
    evaluator = Evaluator(model, test_loader, device)
    
    # Evaluate
    logging.info("Starting evaluation...")
    metrics = evaluator.evaluate()
    
    # Create visualizations
    logging.info("Creating visualizations...")
    visualizer = SentimentVisualizer(save_dir=output_dir)
    
    # Plot confusion matrix
    visualizer.plot_confusion_matrix(
        metrics["confusion_matrix"],
        title="Test Set Confusion Matrix",
        filename="test_confusion_matrix.png",
    )
    
    # Save results
    results = {
        "metrics": metrics,
        "model_path": model_path,
        "data_path": data_path,
        "device": str(device),
    }
    
    results_path = Path(output_dir) / "evaluation_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    logging.info(f"Results saved to {results_path}")
    
    # Print summary
    logging.info("Evaluation Summary:")
    logging.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logging.info(f"Macro F1: {metrics['macro_f1']:.4f}")
    logging.info(f"Weighted F1: {metrics['weighted_f1']:.4f}")
    
    return results


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Evaluate multimodal sentiment analysis model")
    parser.add_argument("--model-path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--data-path", type=str, required=True, help="Path to test data")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--device", type=str, default="auto", help="Device to use")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Evaluate model
    results = evaluate_model(
        model_path=args.model_path,
        data_path=args.data_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        device=args.device,
    )
    
    logging.info("Evaluation completed successfully!")


if __name__ == "__main__":
    main()
