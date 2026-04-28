"""Utility functions for device management and deterministic training."""

import os
import random
from typing import Optional, Union

import numpy as np
import torch
import torch.backends.cudnn as cudnn


def get_device(device: Optional[str] = None) -> torch.device:
    """
    Get the best available device for training.
    
    Priority: CUDA -> MPS (Apple Silicon) -> CPU
    
    Args:
        device: Optional device specification. If None, auto-detect.
        
    Returns:
        torch.device: The selected device.
    """
    if device is None or device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    
    return torch.device(device)


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """
    Set random seeds for reproducible training.
    
    Args:
        seed: Random seed value.
        deterministic: Whether to use deterministic algorithms.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        # Enable deterministic algorithms
        torch.use_deterministic_algorithms(True)
        cudnn.deterministic = True
        cudnn.benchmark = False
        
        # Set environment variables for deterministic behavior
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        os.environ["PYTHONHASHSEED"] = str(seed)


def get_model_size(model: torch.nn.Module) -> dict:
    """
    Calculate model size statistics.
    
    Args:
        model: PyTorch model.
        
    Returns:
        dict: Model size statistics including total parameters, trainable parameters, etc.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Calculate model size in MB
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_all_mb = (param_size + buffer_size) / 1024**2
    
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "non_trainable_parameters": total_params - trainable_params,
        "model_size_mb": size_all_mb,
    }


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model.
        
    Returns:
        int: Number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_number(num: Union[int, float]) -> str:
    """
    Format large numbers with appropriate suffixes (K, M, B).
    
    Args:
        num: Number to format.
        
    Returns:
        str: Formatted number string.
    """
    if num >= 1e9:
        return f"{num/1e9:.1f}B"
    elif num >= 1e6:
        return f"{num/1e6:.1f}M"
    elif num >= 1e3:
        return f"{num/1e3:.1f}K"
    else:
        return str(num)
