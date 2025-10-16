"""
Device management utilities for training.

This module provides functions to handle device selection, model transfer,
and hardware information.
"""

import torch
from typing import Optional, Union


def get_device(device: Optional[str] = None) -> torch.device:
    """
    Get torch device for training.

    Args:
        device: Device specification ("cuda", "cpu", "auto", or None)
                If "auto" or None, automatically select CUDA if available

    Returns:
        torch.device object
    """
    if device is None or device == "auto":
        # Use cpu is cuda is not available
        device = "cuda" if torch.cuda.is_available() else "cpu"

    return torch.device(device)


def to_device(
    data: Union[torch.Tensor, dict, list, tuple],
    device: torch.device
):
    """
    Move data to specified device.

    Handles tensors, dictionaries, lists, and tuples recursively.

    Args:
        data: Data to move (tensor, dict, list, or tuple)
        device: Target device

    Returns:
        Data on the specified device
    """
    if isinstance(data, torch.Tensor):
        return data.to(device)

    # If data is a dictionary, return value in dict format
    elif isinstance(data, dict):
        return {key: to_device(value, device) for key, value in data.items()}

    # If data is a list, return item to device
    elif isinstance(data, list):
        return [to_device(item, device) for item in data]

    # If data is a tuple, return item to device
    elif isinstance(data, tuple):
        return tuple(to_device(item, device) for item in data)
    else:
        return data


def get_device_info(device: Optional[torch.device] = None) -> dict:
    """
    Get information about the device.

    Args:
        device: Device to get info for (defaults to current device)

    Returns:
        Dictionary with device information
    """
    if device is None:
        device = get_device()

    info = {
        "device_type": device.type,
        "device_index": device.index if device.type == "cuda" else None,
        "cuda_available": torch.cuda.is_available(),
    }

    # If CUDA device detected
    if torch.cuda.is_available():
        info.update({
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_current_device": torch.cuda.current_device(),
        })

        if device.type == "cuda":
            device_idx = device.index if device.index is not None else torch.cuda.current_device()
            props = torch.cuda.get_device_properties(device_idx)
            info.update({
                "device_name": props.name,
                "total_memory_gb": props.total_memory / (1024 ** 3), # 1024 ** 3 = 1,073,741,824 approx the amount of bytes in a GB
                "multi_processor_count": props.multi_processor_count,
                "cuda_capability": f"{props.major}.{props.minor}",
            })

    return info


def print_device_info(device: Optional[torch.device] = None):
    """
    Print information about the device.

    Args:
        device: Device to print info for (defaults to current device)
    """
    info = get_device_info(device)

    print("\n" + "=" * 60)
    print("Device Information")
    print("=" * 60)

    print(f"\nDevice type: {info['device_type']}")
    print(f"CUDA available: {info['cuda_available']}")

    if info['cuda_available']:
        print(f"CUDA device count: {info['cuda_device_count']}")
        print(f"Current CUDA device: {info['cuda_current_device']}")

        if info['device_type'] == 'cuda':
            print(f"\nDevice name: {info['device_name']}")
            print(f"Total memory: {info['total_memory_gb']:.2f} GB")
            print(f"Multi-processor count: {info['multi_processor_count']}")
            print(f"CUDA capability: {info['cuda_capability']}")

    print("=" * 60)


def get_memory_info(device: Optional[torch.device] = None) -> dict:
    """
    Get memory usage information for CUDA device.

    Args:
        device: Device to get memory info for

    Returns:
        Dictionary with memory information (in GB)
    """
    if device is None:
        device = get_device()

    if device.type != "cuda":
        return {
            "error": "Memory info only available for CUDA devices"
        }

    device_idx = device.index if device.index is not None else torch.cuda.current_device()

    # 1024 ** 3 = approx the amount of bytes in a GB
    allocated = torch.cuda.memory_allocated(device_idx) / (1024 ** 3)
    reserved = torch.cuda.memory_reserved(device_idx) / (1024 ** 3)
    max_allocated = torch.cuda.max_memory_allocated(device_idx) / (1024 ** 3)
    max_reserved = torch.cuda.max_memory_reserved(device_idx) / (1024 ** 3)

    props = torch.cuda.get_device_properties(device_idx)
    total = props.total_memory / (1024 ** 3)

    return {
        "allocated_gb": allocated,
        "reserved_gb": reserved,
        "max_allocated_gb": max_allocated,
        "max_reserved_gb": max_reserved,
        "total_gb": total,
        "free_gb": total - allocated,
        "utilization_percent": (allocated / total) * 100
    }


def print_memory_info(device: Optional[torch.device] = None):
    """
    Print memory usage information for CUDA device.

    Args:
        device: Device to print memory info for
    """
    info = get_memory_info(device)

    if "error" in info:
        print(f"\n{info['error']}")
        return

    print("\n" + "=" * 60)
    print("GPU Memory Usage")
    print("=" * 60)

    print(f"\nAllocated: {info['allocated_gb']:.2f} GB")
    print(f"Reserved: {info['reserved_gb']:.2f} GB")
    print(f"Max allocated: {info['max_allocated_gb']:.2f} GB")
    print(f"Max reserved: {info['max_reserved_gb']:.2f} GB")
    print(f"Total: {info['total_gb']:.2f} GB")
    print(f"Free: {info['free_gb']:.2f} GB")
    print(f"Utilization: {info['utilization_percent']:.1f}%")

    print("=" * 60)


def clear_memory(device: Optional[torch.device] = None):
    """
    Clear CUDA memory cache.

    Useful for:
    - Freeing up memory between training runs
    - Debugging memory issues
    - Ensuring clean state for memory profiling

    Args:
        device: Device to clear memory for
    """
    if device is None:
        device = get_device()

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)


def reset_peak_memory_stats(device: Optional[torch.device] = None):
    """
    Reset peak memory statistics.

    Useful for profiling memory usage during specific operations.

    Args:
        device: Device to reset stats for
    """
    if device is None:
        device = get_device()

    if device.type == "cuda":
        device_idx = device.index if device.index is not None else torch.cuda.current_device()
        torch.cuda.reset_peak_memory_stats(device_idx)


def set_seed(seed: int, deterministic: bool = False):
    """
    Set random seed for reproducibility.

    Args:
        seed: Random seed
        deterministic: Whether to use deterministic algorithms (slower but more reproducible)
    """
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        # Enable cudnn autotuner for better performance
        torch.backends.cudnn.benchmark = True


__all__ = [
    'get_device',
    'to_device',
    'get_device_info',
    'print_device_info',
    'get_memory_info',
    'print_memory_info',
    'clear_memory',
    'reset_peak_memory_stats',
    'set_seed'
]
