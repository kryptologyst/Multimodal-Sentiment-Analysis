#!/usr/bin/env python3
"""Main training script for multimodal sentiment analysis."""

import argparse
import logging
import os
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from src.data.dataset import create_data_loaders
from src.eval.trainer import Trainer, Evaluator
from src.eval.metrics import MultimodalLoss
from src.models.multimodal_model import MultimodalSentimentModel
from src.utils.device import get_device, set_seed
from src.viz.visualizer import SentimentVisualizer


def setup_logging(log_level: str = "INFO", log_file: str = "training.log") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    import yaml
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    return config


def create_model(config: Dict[str, Any]) -> MultimodalSentimentModel:
    """Create the multimodal sentiment model."""
    model_config = config["model"]
    
    model = MultimodalSentimentModel(
        text_model_name=model_config["text_encoder"]["model_name"],
        text_hidden_dim=model_config["text_encoder"].get("hidden_dim", 256),
        audio_input_dim=model_config["audio_encoder"]["feature_dim"],
        audio_hidden_dim=model_config["audio_encoder"]["hidden_dim"],
        fusion_method=model_config["fusion"]["method"],
        fusion_hidden_dim=model_config["fusion"]["hidden_dim"],
        num_classes=model_config["classifier"]["num_classes"],
        classifier_hidden_dim=model_config["classifier"]["hidden_dim"],
        dropout=model_config["fusion"]["dropout"],
        freeze_text_encoder=model_config["text_encoder"]["freeze_encoder"],
    )
    
    return model


def create_optimizer_and_scheduler(
    model: MultimodalSentimentModel,
    config: Dict[str, Any],
    num_training_steps: int,
) -> tuple:
    """Create optimizer and learning rate scheduler."""
    training_config = config["training"]
    
    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=training_config["learning_rate"],
        weight_decay=training_config["weight_decay"],
    )
    
    # Create scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=training_config["warmup_steps"],
        num_training_steps=num_training_steps,
    )
    
    return optimizer, scheduler


def create_loss_function(config: Dict[str, Any]) -> MultimodalLoss:
    """Create loss function."""
    training_config = config["training"]
    
    loss_fn = MultimodalLoss(
        classification_weight=1.0,
        contrastive_weight=0.1,
        use_focal_loss=False,  # Can be configured
    )
    
    return loss_fn


def train_model(config: Dict[str, Any]) -> Dict[str, Any]:
    """Train the multimodal sentiment model."""
    # Setup
    device = get_device(config["device"]["device"])
    set_seed(config["device"]["seed"], config["device"]["deterministic"])
    
    logging.info(f"Using device: {device}")
    logging.info(f"Configuration: {config}")
    
    # Create data loaders
    data_config = config["data"]
    train_loader, val_loader, test_loader = create_data_loaders(
        data_path=data_config.get("dataset_path", "data"),
        batch_size=config["training"]["batch_size"],
        num_workers=4,
        tokenizer_name=config["model"]["text_encoder"]["model_name"],
        max_text_length=config["model"]["text_encoder"]["max_length"],
        max_audio_length=data_config["max_audio_length"],
        sample_rate=data_config["sample_rate"],
        n_mels=data_config["n_mels"],
        n_fft=data_config["n_fft"],
        hop_length=data_config["hop_length"],
    )
    
    logging.info(f"Train samples: {len(train_loader.dataset)}")
    logging.info(f"Val samples: {len(val_loader.dataset)}")
    logging.info(f"Test samples: {len(test_loader.dataset)}")
    
    # Create model
    model = create_model(config)
    logging.info(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create optimizer and scheduler
    num_training_steps = len(train_loader) * config["training"]["num_epochs"]
    optimizer, scheduler = create_optimizer_and_scheduler(model, config, num_training_steps)
    
    # Create loss function
    criterion = create_loss_function(config)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        save_dir=config["paths"]["checkpoint_dir"],
        use_amp=config["training"]["use_amp"],
    )
    
    # Train
    logging.info("Starting training...")
    training_history = trainer.train(
        num_epochs=config["training"]["num_epochs"],
        patience=config["training"]["patience"],
        min_delta=config["training"]["min_delta"],
    )
    
    # Evaluate
    logging.info("Starting evaluation...")
    evaluator = Evaluator(model, test_loader, device)
    test_metrics = evaluator.evaluate()
    
    # Create visualizations
    logging.info("Creating visualizations...")
    visualizer = SentimentVisualizer(save_dir=config["paths"]["assets_dir"])
    
    # Save visualizations
    visualizer.save_all_visualizations(
        metrics=test_metrics,
        confusion_matrix=test_metrics["confusion_matrix"],
        training_history=training_history,
        labels=[],  # Would need to collect from test set
        predictions=[],  # Would need to collect from test set
        probabilities=[],  # Would need to collect from test set
    )
    
    # Save results
    results = {
        "test_metrics": test_metrics,
        "training_history": training_history,
        "config": config,
    }
    
    results_path = Path(config["paths"]["output_dir"]) / "results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    
    import json
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    logging.info(f"Results saved to {results_path}")
    
    return results


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Train multimodal sentiment analysis model")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    parser.add_argument("--log-file", type=str, default="training.log", help="Log file path")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level, args.log_file)
    
    # Load config
    config = load_config(args.config)
    
    # Train model
    results = train_model(config)
    
    logging.info("Training completed successfully!")


if __name__ == "__main__":
    main()
