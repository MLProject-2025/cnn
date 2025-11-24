"""
Image preprocessing utility with resize and padding.

This script provides functions to resize images while maintaining aspect ratio
using padding. It also includes batch processing capabilities for dataset preparation.
"""

import os
import glob
import numpy as np
import shutil
from PIL import Image
from tqdm import tqdm

# --- Configuration ---
SOURCE_DATA_DIR = './dataset/univ_ML_basic/deepfake/original' 
DEST_DIR = './deepfake_60k_224px'  # Name of folder to be created
TARGET_SIZE = 224  # Target size
TARGET_FAKE_PER_FOLDER = 10000  # Number of samples per Fake folder

# Define image extensions
VALID_EXTENSIONS = ('.jpg',)

def resize_with_pad(img, target_size):
    """
    Resize an image to target_size x target_size while maintaining aspect ratio,
    with padding to center it (black padding).
    """
    # Convert RGBA (transparent) to RGB (prevents save errors)
    if img.mode == 'RGBA':
        img = img.convert('RGB')
        
    w, h = img.size
    
    # Calculate resize maintaining aspect ratio
    scale = target_size / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Resize (LANCZOS: high quality)
    img_resized = img.resize((new_w, new_h), Image.LANCZOS)
    
    # Create black background
    new_img = Image.new("RGB", (target_size, target_size), (0, 0, 0))
    
    # Calculate center coordinates
    paste_x = (target_size - new_w) // 2
    paste_y = (target_size - new_h) // 2
    
    # Paste onto center
    new_img.paste(img_resized, (paste_x, paste_y))
    
    return new_img

def find_all_images(root_dir):
    """Find all images in subdirectories using os.walk"""
    image_list = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(VALID_EXTENSIONS):
                image_list.append(os.path.join(root, file))
    return image_list

def main():
    print(f"=== Starting 60k Data Preprocessing (224px) and Saving ===")
    
    # Initialize destination folder
    if os.path.exists(DEST_DIR):
        print(f"Deleting existing '{DEST_DIR}' folder...")
        shutil.rmtree(DEST_DIR)
    
    os.makedirs(os.path.join(DEST_DIR, 'face_real'))
    os.makedirs(os.path.join(DEST_DIR, 'face_fake'))
    
    # --- 1. Collect image paths ---
    print("\n[Step 1] Collecting and sampling image paths...")
    
    # (1) Real (wiki)
    wiki_path = os.path.join(SOURCE_DATA_DIR, 'wiki')
    real_paths = find_all_images(wiki_path)
    print(f"  - Real (wiki): {len(real_paths)} images found (using all)")
    
    # (2) Fake (3 folders)
    fake_categories = ['inpainting', 'insight', 'text2img']
    fake_copy_list = []  # (source_path, destination_filename)
    
    for cat in fake_categories:
        cat_path = os.path.join(SOURCE_DATA_DIR, cat)
        cat_images = find_all_images(cat_path)
        print(f"  - Fake ({cat}): {len(cat_images)} images found")
        
        # Sampling
        if len(cat_images) >= TARGET_FAKE_PER_FOLDER:
            sampled = np.random.choice(cat_images, TARGET_FAKE_PER_FOLDER, replace=False).tolist()
        else:
            print(f"    ⚠️ {cat}: Less than 10k images -> using all")
            sampled = cat_images
            
        for src_path in sampled:
            filename = os.path.basename(src_path)
            # Prevent filename collisions
            new_filename = f"{cat}_{filename}"
            # Standardize extension to .jpg (optional)
            name_only = os.path.splitext(new_filename)[0]
            new_filename = f"{name_only}.jpg"
            
            fake_copy_list.append((src_path, new_filename))

    total_files = len(real_paths) + len(fake_copy_list)
    print(f"\n  => Total files to process: {total_files} images")

    # --- 2. Process and save images ---
    print("\n[Step 2] Starting resize (224x224) and save...")
    
    # (1) Process Real
    success_count = 0
    error_count = 0
    
    for src_path in tqdm(real_paths, desc="Processing Real"):
        try:
            with Image.open(src_path) as img:
                # Call resize + padding function
                processed_img = resize_with_pad(img, TARGET_SIZE)
                
                # Save path
                filename = os.path.basename(src_path)
                name_only = os.path.splitext(filename)[0]
                save_name = f"{name_only}.jpg"  # standardize to jpg
                dst_path = os.path.join(DEST_DIR, 'face_real', save_name)
                
                # Save (quality=95 for good image quality)
                processed_img.save(dst_path, quality=95)
                success_count += 1
        except Exception as e:
            # print(f"Error processing {src_path}: {e}")
            error_count += 1

    # (2) Process Fake images
    for src_path, save_name in tqdm(fake_copy_list, desc="Processing Fake"):
        try:
            with Image.open(src_path) as img:
                # Resize to 224x224 even if already 512x512
                processed_img = resize_with_pad(img, TARGET_SIZE)
                
                dst_path = os.path.join(DEST_DIR, 'face_fake', save_name)
                processed_img.save(dst_path, quality=95)
                success_count += 1
        except Exception as e:
            error_count += 1

    print(f"\n=== Task Completed! ===")
    print(f"Success: {success_count} images, Failed: {error_count} images")
    print(f"Saved to: {os.path.abspath(DEST_DIR)}")

if __name__ == "__main__":
    main()
