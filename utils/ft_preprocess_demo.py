"""
Fourier Transform preprocessing demonstration script.

This script demonstrates the Fourier Transform preprocessing utilities
for frequency domain analysis of deepfake images.

For working implementations, see:
- experiments/fourier_transform/ft_transform.ipynb
- experiments/fourier_transform/train_googlenet_amplitude_only_ft.ipynb
- experiments/fourier_transform/train_googlenet_phase_only_ft.ipynb
- experiments/fourier_transform_notes.md
"""

import argparse
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser(description='Fourier Transform preprocessing demo')
    parser.add_argument('--image', type=str, required=False,
                        help='Path to input image (optional)')
    parser.add_argument('--ft-type', type=int, default=1, choices=[0, 1],
                        help='Fourier Transform type: 0=phase_only, 1=amplitude_only')
    parser.add_argument('--output', type=str, default='./ft_output.png',
                        help='Output path for transformed image')
    
    args = parser.parse_args()
    
    ft_type_names = ['phase_only', 'amplitude_only']
    
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║          Fourier Transform Preprocessing Demo                ║
╚══════════════════════════════════════════════════════════════╝

Configuration:
  Input Image:  {args.image if args.image else 'None (demo mode)'}
  FT Type:      {args.ft_type} ({ft_type_names[args.ft_type]})
  Output Path:  {args.output}

Available Fourier Transform Types:
  0 - phase_only:      Retains only phase information, amplitude = 1
  1 - amplitude_only:  Removes phase, keeps only amplitude information

About Fourier Transform Preprocessing:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Fourier Transform decomposes images into frequency components:
  • Amplitude: Represents the strength of each frequency
  • Phase: Represents the position/alignment of each frequency

Phase-only mode (ft_type=0):
  - Keeps spatial structure information
  - Removes global illumination and contrast
  - Useful for detecting structural artifacts in deepfakes

Amplitude-only mode (ft_type=1):
  - Visualizes frequency spectrum
  - Shows periodic patterns and textures
  - Useful for detecting GAN fingerprints and compression artifacts

Usage in training:
  from utils.fourier_transform import preprocess_FT
  
  # In your data loader:
  img = Image.open(image_path)
  transformed_img = preprocess_FT(img, Type=1)  # amplitude_only

Working Examples:
  - experiments/fourier_transform/ft_transform.ipynb
  - experiments/fourier_transform/train_googlenet_amplitude_only_ft.ipynb
  - experiments/fourier_transform/train_googlenet_phase_only_ft.ipynb
  - experiments/finetuning/fft/fft_resnet50_finetuning.ipynb

For more details, see experiments/fourier_transform_notes.md
""")
    
    # Try to import and run the actual function if dependencies are available
    try:
        from utils.fourier_transform import preprocess_FT, demo
        print("\n" + "="*62)
        print("Running Fourier Transform utility demo...")
        print("="*62 + "\n")
        demo()
        
        if args.image:
            from PIL import Image
            img = Image.open(args.image)
            result = preprocess_FT(img, Type=args.ft_type)
            result.save(args.output)
            print(f"\n✓ Transformed image saved to: {args.output}")
    except ImportError as e:
        print(f"\n⚠ Note: Dependencies not installed. Install with: pip install -r requirements.txt")
        print(f"   Error: {e}")


if __name__ == '__main__':
    main()
