"""
Dataset analysis and statistics utility for deepfake detection.

This script analyzes the dataset to provide statistics on file counts,
file types, and image dimensions.
"""

import os
import glob
import random
import numpy as np
from collections import Counter
from PIL import Image
from tqdm import tqdm

# --- Configuration ---
DATA_ROOT = './dataset/univ_ML_basic/deepfake/original'  # Root folder containing the dataset
SAMPLE_SIZE_FOR_SIZE_CHECK = 1000  # Number of images to sample for size analysis

# Image file extensions to consider
IMAGE_EXTS = {'.jpg'}

def analyze_folder(folder_path):
    """
    Returns file counts, extension distribution, and image size statistics for a folder.
    """
    if not os.path.exists(folder_path):
        return None

    total_files = 0
    extension_counts = Counter()
    all_image_paths = []

    # 1. Scan files (recursively search with os.walk)
    print(f"   📂 Scanning... '{os.path.basename(folder_path)}'")
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            total_files += 1
            ext = os.path.splitext(file)[1].lower()
            extension_counts[ext] += 1
            
            if ext in IMAGE_EXTS:
                all_image_paths.append(os.path.join(root, file))
    
    # 2. Image size analysis (sampling)
    width_stats = {'min': 0, 'max': 0, 'mean': 0}
    height_stats = {'min': 0, 'max': 0, 'mean': 0}
    
    if all_image_paths:
        # Sampling (use all if total count is less than sample size)
        if len(all_image_paths) > SAMPLE_SIZE_FOR_SIZE_CHECK:
            sampled_paths = random.sample(all_image_paths, SAMPLE_SIZE_FOR_SIZE_CHECK)
        else:
            sampled_paths = all_image_paths
            
        widths = []
        heights = []
        
        for img_path in tqdm(sampled_paths, desc=f"   📏 Measuring sizes ({len(sampled_paths)} images)", leave=False):
            try:
                with Image.open(img_path) as img:
                    w, h = img.size
                    widths.append(w)
                    heights.append(h)
            except Exception:
                pass  # Ignore corrupted images
        
        if widths:
            widths = np.array(widths)
            heights = np.array(heights)
            width_stats = {'min': widths.min(), 'max': widths.max(), 'mean': widths.mean()}
            height_stats = {'min': heights.min(), 'max': heights.max(), 'mean': heights.mean()}

    return {
        'total': total_files,
        'exts': extension_counts,
        'w_stats': width_stats,
        'h_stats': height_stats,
        'img_count': len(all_image_paths)  # actual number of image files
    }

def main():
    print(f"=== Dataset Detailed Analysis (Count + Size) ===")
    print(f"Target path: {os.path.abspath(DATA_ROOT)}\n")

    if not os.path.exists(DATA_ROOT):
        print(f"❌ Error: '{DATA_ROOT}' folder not found.")
        return

    try:
        subfolders = [f for f in os.listdir(DATA_ROOT) if os.path.isdir(os.path.join(DATA_ROOT, f))]
        subfolders.sort()
    except Exception as e:
        print(f"Error: {e}")
        return

    if not subfolders:
        print("❌ No subfolders found.")
        return

    print(f"{'Folder Name':<12} | {'Files':<9} | {'Images':<9} | {'Avg Size (WxH)':<18} | {'Extensions'}")
    print("-" * 95)

    total_images_sum = 0

    for folder in subfolders:
        folder_path = os.path.join(DATA_ROOT, folder)
        result = analyze_folder(folder_path)
        
        if result is None:
            print(f"{folder:<12} | {'Path N/A':<9} |")
            continue
            
        # Format results
        count_str = f"{result['total']:,}"
        img_count_str = f"{result['img_count']:,}"
        
        w_mean = result['w_stats']['mean']
        h_mean = result['h_stats']['mean']
        size_str = f"{w_mean:.0f}x{h_mean:.0f}" if w_mean > 0 else "N/A"
        
        # Show only major extensions (top 3)
        top_exts = result['exts'].most_common(3)
        ext_str = ", ".join([f"{k} {v}" for k, v in top_exts])
        
        print(f"{folder:<12} | {count_str:<9} | {img_count_str:<9} | {size_str:<18} | {ext_str}")
        
        total_images_sum += result['img_count']

    print("-" * 95)
    print(f"Total image files: {total_images_sum:,} images")
    print("※ Average size is based on sampling up to 1,000 images per folder.")

if __name__ == "__main__":
    main()
