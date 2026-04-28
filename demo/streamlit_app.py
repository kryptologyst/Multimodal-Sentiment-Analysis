"""Interactive Streamlit demo for multimodal sentiment analysis."""

import os
import tempfile
from pathlib import Path
from typing import Dict, Optional

import librosa
import numpy as np
import streamlit as st
import torch
from transformers import AutoTokenizer

# Add src to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.models.multimodal_model import MultimodalSentimentModel
from src.utils.device import get_device, set_seed
from src.viz.visualizer import visualize_audio_features


# Page configuration
st.set_page_config(
    page_title="Multimodal Sentiment Analysis",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #667eea;
    }
    .sentiment-positive {
        color: #2E8B57;
        font-weight: bold;
    }
    .sentiment-negative {
        color: #DC143C;
        font-weight: bold;
    }
    .sentiment-neutral {
        color: #FFD700;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model(model_path: Optional[str] = None) -> MultimodalSentimentModel:
    """Load the trained model."""
    if model_path and os.path.exists(model_path):
        # Load from checkpoint
        checkpoint = torch.load(model_path, map_location="cpu")
        model = MultimodalSentimentModel()
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        # Create a new model for demo
        model = MultimodalSentimentModel()
    
    device = get_device()
    model.to(device)
    model.eval()
    
    return model


@st.cache_resource
def load_tokenizer(model_name: str = "distilbert-base-uncased"):
    """Load the tokenizer."""
    return AutoTokenizer.from_pretrained(model_name)


def preprocess_audio(audio_file) -> torch.Tensor:
    """Preprocess uploaded audio file."""
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(audio_file.read())
        tmp_path = tmp_file.name
    
    try:
        # Load audio
        y, sr = librosa.load(tmp_path, sr=16000)
        
        # Truncate or pad to 10 seconds
        max_length = 10 * sr
        if len(y) > max_length:
            y = y[:max_length]
        else:
            y = np.pad(y, (0, max_length - len(y)), mode="constant")
        
        # Extract mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_mels=80,
            n_fft=1024,
            hop_length=256,
        )
        
        # Convert to log scale
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Convert to tensor
        audio_tensor = torch.from_numpy(log_mel_spec).float()
        
        return audio_tensor, y, sr
        
    finally:
        # Clean up temporary file
        os.unlink(tmp_path)


def predict_sentiment(
    model: MultimodalSentimentModel,
    tokenizer,
    text: str,
    audio_tensor: torch.Tensor,
    device: torch.device,
) -> Dict[str, float]:
    """Predict sentiment for given text and audio."""
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
    audio = audio_tensor.unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(text_input_ids, text_attention_mask, audio)
        probabilities = torch.softmax(outputs["logits"], dim=1)
        prediction = torch.argmax(probabilities, dim=1)
    
    # Convert to probabilities
    probs = probabilities.cpu().numpy()[0]
    class_names = ["positive", "negative", "neutral"]
    
    return {
        "prediction": class_names[prediction.item()],
        "probabilities": {
            class_name: float(prob)
            for class_name, prob in zip(class_names, probs)
        },
    }


def main():
    """Main Streamlit app."""
    # Header
    st.markdown('<h1 class="main-header">🎭 Multimodal Sentiment Analysis</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Configuration")
    
    # Model selection
    model_option = st.sidebar.selectbox(
        "Model",
        ["Demo Model", "Custom Model"],
        help="Choose between demo model or load a custom trained model"
    )
    
    model_path = None
    if model_option == "Custom Model":
        model_path = st.sidebar.file_uploader(
            "Upload Model Checkpoint",
            type=["pt", "pth"],
            help="Upload a trained model checkpoint"
        )
    
    # Load model and tokenizer
    if model_path:
        model = load_model(model_path)
    else:
        model = load_model()
    
    tokenizer = load_tokenizer()
    device = get_device()
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📝 Text Input")
        text_input = st.text_area(
            "Enter your text here:",
            value="I absolutely love this product! It's amazing and works perfectly.",
            height=100,
            help="Enter the text you want to analyze for sentiment"
        )
    
    with col2:
        st.header("🎵 Audio Input")
        audio_file = st.file_uploader(
            "Upload Audio File",
            type=["wav", "mp3", "flac", "m4a"],
            help="Upload an audio file to analyze sentiment from voice tone"
        )
        
        if audio_file is not None:
            st.audio(audio_file, format="audio/wav")
    
    # Process button
    if st.button("🔍 Analyze Sentiment", type="primary"):
        if not text_input.strip():
            st.error("Please enter some text to analyze.")
            return
        
        # Process audio if provided
        if audio_file is not None:
            try:
                audio_tensor, y, sr = preprocess_audio(audio_file)
            except Exception as e:
                st.error(f"Error processing audio: {str(e)}")
                return
        else:
            # Generate dummy audio for demo
            audio_tensor = torch.randn(80, 625)  # Dummy mel spectrogram
        
        # Predict sentiment
        try:
            result = predict_sentiment(model, tokenizer, text_input, audio_tensor, device)
            
            # Display results
            st.header("📊 Results")
            
            # Prediction
            prediction = result["prediction"]
            probabilities = result["probabilities"]
            
            # Color coding for sentiment
            if prediction == "positive":
                sentiment_class = "sentiment-positive"
                emoji = "😊"
            elif prediction == "negative":
                sentiment_class = "sentiment-negative"
                emoji = "😞"
            else:
                sentiment_class = "sentiment-neutral"
                emoji = "😐"
            
            st.markdown(f"### {emoji} Predicted Sentiment: <span class='{sentiment_class}'>{prediction.upper()}</span>", unsafe_allow_html=True)
            
            # Probability bars
            st.subheader("Confidence Scores")
            
            for sentiment, prob in probabilities.items():
                if sentiment == "positive":
                    color = "#2E8B57"
                elif sentiment == "negative":
                    color = "#DC143C"
                else:
                    color = "#FFD700"
                
                st.progress(prob, text=f"{sentiment.title()}: {prob:.2%}")
            
            # Detailed metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Positive", f"{probabilities['positive']:.2%}")
            with col2:
                st.metric("Negative", f"{probabilities['negative']:.2%}")
            with col3:
                st.metric("Neutral", f"{probabilities['neutral']:.2%}")
            
            # Audio visualization if provided
            if audio_file is not None:
                st.subheader("🎵 Audio Analysis")
                
                # Create audio visualization
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                    tmp_file.write(audio_file.read())
                    tmp_path = tmp_file.name
                
                try:
                    fig = visualize_audio_features(tmp_path, save_dir="temp")
                    st.pyplot(fig)
                finally:
                    os.unlink(tmp_path)
            
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
    
    # Information section
    st.header("ℹ️ About")
    
    st.markdown("""
    This demo showcases **Multimodal Sentiment Analysis** that combines:
    
    - **Text Analysis**: Uses DistilBERT to analyze textual sentiment
    - **Audio Analysis**: Extracts mel spectrogram features from audio
    - **Fusion**: Combines text and audio features using various fusion strategies
    
    ### Features:
    - 🎯 **Late Fusion**: Concatenates text and audio embeddings
    - 🎯 **Cross-Attention**: Uses attention mechanisms for fusion
    - 🎯 **Contrastive Learning**: Aligns text and audio representations
    - 🎯 **Real-time Analysis**: Process text and audio in real-time
    
    ### Model Architecture:
    - **Text Encoder**: DistilBERT (pre-trained transformer)
    - **Audio Encoder**: CNN + LSTM for mel spectrogram processing
    - **Fusion Module**: Configurable fusion strategies
    - **Classifier**: Multi-layer perceptron for sentiment classification
    """)
    
    # Safety disclaimer
    st.markdown("---")
    st.markdown("""
    ### ⚠️ Disclaimer
    This is a research/educational tool for demonstration purposes. 
    The model may not be suitable for production use without proper validation and testing.
    """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        Multimodal Sentiment Analysis Demo | Built with Streamlit & PyTorch
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
