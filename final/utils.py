import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
import os
import cv2
import numpy as np
import random
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix
import glob
from PIL import Image

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 화질 저하, 블러, 노이즈 추가 함수
def strong_aug(img_np):
    if random.random() < 0.5: # JPEG Compression
        q = random.randint(30, 90)
        _, enc = cv2.imencode('.jpg', img_np, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        img_np = cv2.imdecode(enc, 1)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

    if random.random() < 0.3: # Blur
        img_np = cv2.GaussianBlur(img_np, (5, 5), 0)
        
    if random.random() < 0.3: # Noise
        noise = np.random.normal(0, 15, img_np.shape).astype(np.uint8)
        img_np = np.clip(img_np.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
    return img_np

# Mixup 함수
def mixup_data(x, y, alpha=1.0):
    if alpha > 0: lam = np.random.beta(alpha, alpha)
    else: lam = 1
    index = torch.randperm(x.size(0)).to(DEVICE)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# 디렉토리에서 'real', 'fake' 폴더 찾는 함수
def find_class_dirs(root_dir):

    print(f"Scanning: {os.path.abspath(root_dir)}") # 절대 경로 확인용 출력
    
    if not os.path.exists(root_dir):
        print(f"[Error] 경로가 존재하지 않습니다: {root_dir}")
        print("  -> 현재 작업 경로(os.getcwd()):", os.getcwd())
        return None, None

    real_dir, fake_dir = None, None
    
    # 1. 루트 바로 아래 탐색
    subdirs = [os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    
    # 2. 만약 바로 아래에 없다면 1 depth 더 들어가서 탐색 (예: Train/Train/Real...)
    if len(subdirs) == 1:
        nested_root = subdirs[0]
        print(f"  -> Checking nested dir: {nested_root}")
        subdirs = [os.path.join(nested_root, d) for d in os.listdir(nested_root) if os.path.isdir(os.path.join(nested_root, d))]

    for d in subdirs:
        name = os.path.basename(d).lower()
        if 'real' in name: real_dir = d
        elif 'fake' in name: fake_dir = d
            
    return real_dir, fake_dir

# RAM에 데이터 로드 함수
def load_data_to_ram(root_dir, target_size=224, n_samples=None):
    print(f"Loading data from {root_dir} into RAM...")
    
    real_dir, fake_dir = find_class_dirs(root_dir)
    if not real_dir or not fake_dir:
        print(f"[Error] '{root_dir}'에서 Real/Fake 폴더를 못 찾았습니다.")
        return None, None

    print(f" - Real: {os.path.basename(real_dir)}")
    print(f" - Fake: {os.path.basename(fake_dir)}")

    # 1. 모든 경로 수집
    all_files = []
    for label, folder in enumerate([real_dir, fake_dir]): # 0: Real, 1: Fake
        files = glob.glob(os.path.join(folder, "*.*"))
        # 이미지 확장자 필터링
        files = [f for f in files if f.lower().endswith(('.jpg', '.png', '.jpeg', '.webp'))]
        for f in files:
            all_files.append((f, label))
            
    total_files = len(all_files)
    print(f" - Total Images Found: {total_files}")
    
    # 2. 샘플링 (n_samples 설정 시)
    if n_samples is not None and n_samples < total_files:
        print(f" - Sampling {n_samples} images...")
        random.shuffle(all_files)
        all_files = all_files[:n_samples]
    else:
        # 셔플은 학습 데이터 분포를 섞어주므로 기본적으로 해두는 게 좋음
        random.shuffle(all_files)

    # 3. 로딩 시작
    crop_transform = transforms.CenterCrop(target_size)
    resize_transform = transforms.Resize((target_size, target_size))
    to_uint8_tensor = transforms.PILToTensor() 

    error_count = 0 # 에러 카운트 추가
    
    tensor_list = []
    label_list = []
    
    for fpath, label in tqdm(all_files, desc="Loading to RAM", leave=False):
        try:
            img = Image.open(fpath).convert('RGB')
            w, h = img.size
            
            # 224보다 크거나 같으면 CenterCrop (화질 보존)
            if w >= target_size and h >= target_size:
                img = crop_transform(img)
            else:
                # 작으면 Resize
                img = resize_transform(img)
            
            # uint8 Tensor 변환
            t_img = to_uint8_tensor(img)
            
            tensor_list.append(t_img)
            label_list.append(label)
        except Exception as e:
             # [디버깅] 처음 10개 에러만 출력
            if error_count < 10:
                print(f"[Error] Failed to load {fpath}: {e}")
                error_count += 1
            pass 


    if not tensor_list:
        print("[Error] 이미지가 없습니다.")
        return None, None

    # Stack
    data_tensor = torch.stack(tensor_list)
    label_tensor = torch.tensor(label_list, dtype=torch.long)
    
    mem_size = data_tensor.element_size() * data_tensor.nelement() / (1024**3)
    print(f"Load Complete! {len(data_tensor)} images. RAM: {mem_size:.2f} GB")
    
    return data_tensor, label_tensor

# dataset 클래스
class RAMDataset(Dataset):
    def __init__(self, data_tensor, label_tensor, exp_name, is_train=True):
        self.data = data_tensor   
        self.labels = label_tensor
        self.exp_name = exp_name
        self.is_train = is_train
        self.normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.to_float = transforms.ToTensor() 

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        img_tensor = self.data[idx]
        label = self.labels[idx]
        
        # Tensor -> Numpy (H, W, 3) for OpenCV
        img_np = img_tensor.permute(1, 2, 0).numpy()
        
        # Augmentation (Train)
        if self.is_train and self.exp_name in ['ROBUST', 'FREQ']:
            img_np = strong_aug(img_np)
        
        # Freq Conversion
        if self.exp_name == 'FREQ':
            img_np = get_freq_feature(img_np)
            
        # Numpy -> Float Tensor & Normalize
        img_out = self.to_float(img_np)
        img_out = self.normalize(img_out)
        
        return img_out, label
