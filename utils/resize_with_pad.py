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
    print(f"=== 6만개 데이터 전처리(224px) 및 저장 시작 ===")
    
    # 목적지 폴더 초기화
    if os.path.exists(DEST_DIR):
        print(f"기존 '{DEST_DIR}' 폴더 삭제 중...")
        shutil.rmtree(DEST_DIR)
    
    os.makedirs(os.path.join(DEST_DIR, 'face_real'))
    os.makedirs(os.path.join(DEST_DIR, 'face_fake'))
    
    # --- 1. 이미지 경로 수집 ---
    print("\n[1단계] 이미지 경로 수집 및 샘플링...")
    
    # (1) Real (wiki)
    wiki_path = os.path.join(SOURCE_DATA_DIR, 'wiki')
    real_paths = find_all_images(wiki_path)
    print(f"  - Real (wiki): {len(real_paths)}장 발견 (전체 사용)")
    
    # (2) Fake (3개 폴더)
    fake_categories = ['inpainting', 'insight', 'text2img']
    fake_copy_list = [] # (원본경로, 저장될파일명)
    
    for cat in fake_categories:
        cat_path = os.path.join(SOURCE_DATA_DIR, cat)
        cat_images = find_all_images(cat_path)
        print(f"  - Fake ({cat}): {len(cat_images)}장 발견")
        
        # 샘플링
        if len(cat_images) >= TARGET_FAKE_PER_FOLDER:
            sampled = np.random.choice(cat_images, TARGET_FAKE_PER_FOLDER, replace=False).tolist()
        else:
            print(f"    ⚠️ {cat}: 1만장 부족 -> 전체 사용")
            sampled = cat_images
            
        for src_path in sampled:
            filename = os.path.basename(src_path)
            # 파일명 충돌 방지
            new_filename = f"{cat}_{filename}"
            # 확장자를 .jpg로 통일 (선택사항)
            name_only = os.path.splitext(new_filename)[0]
            new_filename = f"{name_only}.jpg"
            
            fake_copy_list.append((src_path, new_filename))

    total_files = len(real_paths) + len(fake_copy_list)
    print(f"\n  => 총 처리할 파일 수: {total_files}장")

    # --- 2. 이미지 처리 및 저장 ---
    print("\n[2단계] 리사이징(224x224) 및 저장 시작...")
    
    # (1) Real 처리
    success_count = 0
    error_count = 0
    
    for src_path in tqdm(real_paths, desc="Processing Real"):
        try:
            with Image.open(src_path) as img:
                # 리사이즈 + 패딩 함수 호출
                processed_img = resize_with_pad(img, TARGET_SIZE)
                
                # 저장 경로
                filename = os.path.basename(src_path)
                name_only = os.path.splitext(filename)[0]
                save_name = f"{name_only}.jpg" # jpg로 통일
                dst_path = os.path.join(DEST_DIR, 'face_real', save_name)
                
                # 저장 (압축률 90 정도면 화질 좋음)
                processed_img.save(dst_path, quality=95)
                success_count += 1
        except Exception as e:
            # print(f"Error processing {src_path}: {e}")
            error_count += 1

    # (2) Fake 처리
    for src_path, save_name in tqdm(fake_copy_list, desc="Processing Fake"):
        try:
            with Image.open(src_path) as img:
                # 이미 512x512라도 224x224로 줄임
                processed_img = resize_with_pad(img, TARGET_SIZE)
                
                dst_path = os.path.join(DEST_DIR, 'face_fake', save_name)
                processed_img.save(dst_path, quality=95)
                success_count += 1
        except Exception as e:
            error_count += 1

    print(f"\n=== 작업 완료! ===")
    print(f"성공: {success_count}장, 실패: {error_count}장")
    print(f"저장 위치: {os.path.abspath(DEST_DIR)}")

if __name__ == "__main__":
    main()
