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

# --- 설정 ---
DATA_ROOT = './dataset/univ_ML_basic/deepfake/original' # 데이터가 있는 루트 폴더
SAMPLE_SIZE_FOR_SIZE_CHECK = 1000 # 크기 분석을 위해 몇 장을 샘플링할지

# 이미지로 간주할 확장자들
IMAGE_EXTS = {'.jpg'}

def analyze_folder(folder_path):
    """
    폴더 내의 파일 개수, 확장자 분포, 그리고 이미지 크기 통계를 반환합니다.
    """
    if not os.path.exists(folder_path):
        return None

    total_files = 0
    extension_counts = Counter()
    all_image_paths = []

    # 1. 파일 스캔 (os.walk로 구석구석 찾기)
    print(f"   📂 스캔 중... '{os.path.basename(folder_path)}'")
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            total_files += 1
            ext = os.path.splitext(file)[1].lower()
            extension_counts[ext] += 1
            
            if ext in IMAGE_EXTS:
                all_image_paths.append(os.path.join(root, file))
    
    # 2. 이미지 크기 분석 (샘플링)
    width_stats = {'min': 0, 'max': 0, 'mean': 0}
    height_stats = {'min': 0, 'max': 0, 'mean': 0}
    
    if all_image_paths:
        # 샘플링 (전체 개수가 샘플 수보다 적으면 전체 사용)
        if len(all_image_paths) > SAMPLE_SIZE_FOR_SIZE_CHECK:
            sampled_paths = random.sample(all_image_paths, SAMPLE_SIZE_FOR_SIZE_CHECK)
        else:
            sampled_paths = all_image_paths
            
        widths = []
        heights = []
        
        for img_path in tqdm(sampled_paths, desc=f"   📏 크기 측정 중 ({len(sampled_paths)}장)", leave=False):
            try:
                with Image.open(img_path) as img:
                    w, h = img.size
                    widths.append(w)
                    heights.append(h)
            except Exception:
                pass # 깨진 이미지는 무시
        
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
        'img_count': len(all_image_paths) # 실제 이미지 파일 수
    }

def main():
    print(f"=== 데이터셋 정밀 분석 (개수 + 크기) ===")
    print(f"대상 경로: {os.path.abspath(DATA_ROOT)}\n")

    if not os.path.exists(DATA_ROOT):
        print(f"❌ 오류: '{DATA_ROOT}' 폴더가 없습니다.")
        return

    try:
        subfolders = [f for f in os.listdir(DATA_ROOT) if os.path.isdir(os.path.join(DATA_ROOT, f))]
        subfolders.sort()
    except Exception as e:
        print(f"오류: {e}")
        return

    if not subfolders:
        print("❌ 하위 폴더가 없습니다.")
        return

    print(f"{'폴더명':<12} | {'파일 수':<9} | {'이미지 수':<9} | {'평균 크기 (WxH)':<18} | {'확장자 분포'}")
    print("-" * 95)

    total_images_sum = 0

    for folder in subfolders:
        folder_path = os.path.join(DATA_ROOT, folder)
        result = analyze_folder(folder_path)
        
        if result is None:
            print(f"{folder:<12} | {'경로 없음':<9} |")
            continue
            
        # 결과 포매팅
        count_str = f"{result['total']:,}"
        img_count_str = f"{result['img_count']:,}"
        
        w_mean = result['w_stats']['mean']
        h_mean = result['h_stats']['mean']
        size_str = f"{w_mean:.0f}x{h_mean:.0f}" if w_mean > 0 else "N/A"
        
        # 주요 확장자만 표시 (상위 3개)
        top_exts = result['exts'].most_common(3)
        ext_str = ", ".join([f"{k} {v}" for k, v in top_exts])
        
        print(f"{folder:<12} | {count_str:<9} | {img_count_str:<9} | {size_str:<18} | {ext_str}")
        
        total_images_sum += result['img_count']

    print("-" * 95)
    print(f"총 이미지 파일 합계: {total_images_sum:,} 장")
    print("※ 평균 크기는 폴더별 최대 1,000장 샘플링 기준입니다.")

if __name__ == "__main__":
    main()
