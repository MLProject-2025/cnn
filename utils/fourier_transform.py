"""
Fourier Transform preprocessing utilities for deepfake detection.

This module provides functions to apply various Fourier Transform-based
preprocessing techniques to images for frequency domain analysis.
"""

import numpy as np
from PIL import Image


# Available Fourier Transform types
FT_type = [
    'phase_only',               # 0: Keep only phase information
    'amplitude_only',           # 1: Keep only amplitude information
    'shuffle_global_amplitude', # 2: Shuffle global amplitude (not fully implemented)
    'shuffle_patch_amplitude',  # 3: Shuffle patch amplitude (not fully implemented)
    'average_amplitude',        # 4: Average amplitude (not fully implemented)
    'shuffle_patch_phase'       # 5: Shuffle patch phase (not fully implemented)
]


def preprocess_FT(img, Type=None):
    """
    Apply Fourier Transform preprocessing to an image.
    
    Args:
        img: PIL Image or numpy array
        Type: Integer index of FT_type (0=phase_only, 1=amplitude_only)
    
    Returns:
        PIL Image with Fourier Transform preprocessing applied
    """
    if Type is None:
        return img
        
    Type = FT_type[Type]
    
    # Convert to numpy array if needed
    if isinstance(img, Image.Image):
        img_np = np.array(img)
    else:
        img_np = img
    
    ifft_channels = []
    for i in range(img_np.shape[2]):  # Iterate over channels
        fft_result = np.fft.fft2(img_np[:, :, i])
        
        # --------- Various FT processing methods ---------
        if Type == 'phase_only':  # Keep only phase information
            left_frequency = np.exp(1j * np.angle(fft_result)) 
            ifft_result = np.fft.ifft2(left_frequency)
            ifft_result = np.real(ifft_result)
            
        elif Type == 'amplitude_only':  # Keep only amplitude information
            fft_shifted = np.fft.fftshift(fft_result)        
            amplitude_spectrum = np.log(np.abs(fft_shifted) + 1e-8)
            ifft_result = amplitude_spectrum
            
        else:
            # For unimplemented types, return the original
            ifft_result = np.real(np.fft.ifft2(fft_result))
        
        # Scale to 0-255 range
        ifft_result = _scale_to_uint8(ifft_result)
        ifft_channels.append(ifft_result)
    
    # Stack channels back together
    ifft_image = np.stack(ifft_channels, axis=-1)
    
    return Image.fromarray(ifft_image.astype(np.uint8))


def _scale_to_uint8(img_np):
    """
    Scale numpy array to 0-255 range.
    
    Args:
        img_np: Numpy array
    
    Returns:
        Scaled numpy array (0-255 range)
    """
    min_val, max_val = np.min(img_np), np.max(img_np)
    scaled_np = 255.0 * (img_np - min_val) / (max_val - min_val + 1e-9)
    return scaled_np


def demo():
    """
    Demonstration of Fourier Transform preprocessing.
    
    Usage:
        python utils/fourier_transform.py
    """
    # Create a simple test image
    print("Fourier Transform Preprocessing Demo")
    print("=" * 50)
    print("\nAvailable FT types:")
    for i, ft_type in enumerate(FT_type):
        print(f"  {i}: {ft_type}")
    
    print("\nTo use in your code:")
    print("  from utils.fourier_transform import preprocess_FT")
    print("  transformed_img = preprocess_FT(your_image, Type=0)  # phase_only")
    print("  transformed_img = preprocess_FT(your_image, Type=1)  # amplitude_only")


if __name__ == "__main__":
    demo()
