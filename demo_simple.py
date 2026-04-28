#!/usr/bin/env python3
"""Simple demo script for multimodal sentiment analysis."""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

import torch
from transformers import AutoTokenizer

from src.models.multimodal_model import MultimodalSentimentModel
from src.utils.device import get_device, set_seed


def demo_sentiment_analysis():
    """Demonstrate multimodal sentiment analysis."""
    print("🎭 Multimodal Sentiment Analysis Demo")
    print("=" * 50)
    
    # Setup
    device = get_device()
    set_seed(42)
    
    print(f"Using device: {device}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    
    # Create model
    model = MultimodalSentimentModel()
    model.to(device)
    model.eval()
    
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Demo samples
    samples = [
        {
            "text": "I absolutely love this product! It's amazing and works perfectly.",
            "sentiment": "positive"
        },
        {
            "text": "This is terrible. I hate it and it doesn't work at all.",
            "sentiment": "negative"
        },
        {
            "text": "It's okay, nothing special but it works.",
            "sentiment": "neutral"
        },
        {
            "text": "Fantastic! This exceeded all my expectations.",
            "sentiment": "positive"
        },
        {
            "text": "I'm disappointed with the quality and service.",
            "sentiment": "negative"
        },
    ]
    
    print("\nAnalyzing sentiment for demo samples:")
    print("-" * 50)
    
    correct_predictions = 0
    total_predictions = len(samples)
    
    for i, sample in enumerate(samples, 1):
        text = sample["text"]
        true_sentiment = sample["sentiment"]
        
        # Preprocess text
        text_encoding = tokenizer(
            text,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        text_input_ids = text_encoding["input_ids"].to(device)
        text_attention_mask = text_encoding["attention_mask"].to(device)
        
        # Generate dummy audio (in real usage, this would be actual audio)
        audio = torch.randn(1, 80, 625).to(device)  # Dummy mel spectrogram
        
        # Predict
        with torch.no_grad():
            outputs = model(text_input_ids, text_attention_mask, audio)
            probabilities = torch.softmax(outputs["logits"], dim=1)
            prediction = torch.argmax(probabilities, dim=1)
        
        # Convert to probabilities
        probs = probabilities.cpu().numpy()[0]
        class_names = ["positive", "negative", "neutral"]
        predicted_sentiment = class_names[prediction.item()]
        
        # Check if prediction is correct
        is_correct = predicted_sentiment == true_sentiment
        if is_correct:
            correct_predictions += 1
        
        # Display results
        print(f"\nSample {i}:")
        print(f"Text: {text}")
        print(f"True Sentiment: {true_sentiment}")
        print(f"Predicted Sentiment: {predicted_sentiment}")
        print(f"Confidence: {probs[prediction.item()]:.2%}")
        print(f"Correct: {'✅' if is_correct else '❌'}")
        
        # Show probability distribution
        print("Probability Distribution:")
        for class_name, prob in zip(class_names, probs):
            print(f"  {class_name}: {prob:.2%}")
    
    # Summary
    accuracy = correct_predictions / total_predictions
    print(f"\n" + "=" * 50)
    print(f"Demo Summary:")
    print(f"Total Samples: {total_predictions}")
    print(f"Correct Predictions: {correct_predictions}")
    print(f"Accuracy: {accuracy:.2%}")
    
    print(f"\nModel Architecture:")
    print(f"- Text Encoder: DistilBERT")
    print(f"- Audio Encoder: CNN + LSTM")
    print(f"- Fusion: Late Fusion")
    print(f"- Classifier: MLP")
    
    print(f"\nNote: This demo uses dummy audio data.")
    print(f"For real audio analysis, use the Streamlit demo:")
    print(f"streamlit run demo/streamlit_app.py")


if __name__ == "__main__":
    demo_sentiment_analysis()
