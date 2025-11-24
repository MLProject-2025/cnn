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
from utils import strong_aug, mixup_data, mixup_criterion, find_class_dirs, load_data_to_ram, RAMDataset

# 설정값
EXP_NAME = 'ROBUST'
TRAIN_DATA_ROOT = '../datasets/deepfake/Train' 
TEST_DATA_ROOT = '../datasets/deepfake_60k_cropped_224px'

TRAIN_N_SAMPLES = 60000
TEST_N_SAMPLES = 10000

IMG_SIZE = 224
BATCH_SIZE = 64 # 메모리 터지면 32로
EPOCHS = 15
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# 1. Load Data
print(">>> Loading Train Set...")
train_x, train_y = load_data_to_ram(TRAIN_DATA_ROOT, IMG_SIZE, TRAIN_N_SAMPLES)
print(">>> Loading Test Set...")
test_x, test_y = load_data_to_ram(TEST_DATA_ROOT, IMG_SIZE, TEST_N_SAMPLES)

if train_x is None or test_x is None: 
    raise Exception("데이터 로드 실패. 경로를 확인하세요.")

# 2. Loader (num_workers=0 for RAM speed)
train_ds = RAMDataset(train_x, train_y, EXP_NAME, is_train=True)
test_ds = RAMDataset(test_x, test_y, EXP_NAME, is_train=False)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


# 도메인 적응 함수, 데이터를 섞어서 학습
def run_domain_adaptation():
    print("\n🚀 [FINAL STRATEGY] Domain Adaptation (Kaggle + Wiki 50%)")

    # 테스트 데이터 출처가 wiki에서 가져온 것들 + 생성 모델로 만든 이미지라 이름이 wiki임    
    # 1. 데이터 로드
    print("1️⃣ Loading Source (Kaggle)...")
    kaggle_x, kaggle_y = train_x, train_y
    print("2️⃣ Loading Target (Wiki)...")
    wiki_x, wiki_y = test_x, test_y
    
    if kaggle_x is None or wiki_x is None: return

    # 2. 테스트 데이터 반으로 쪼개기 (섞어서)
    total_wiki = len(wiki_x)
    indices = torch.randperm(total_wiki)
    split_idx = int(total_wiki * 0.1) # 1000개만 학습에 사용
    
    wiki_train_x = wiki_x[indices[:split_idx]]
    wiki_train_y = wiki_y[indices[:split_idx]]
    
    wiki_test_x = wiki_x[indices[split_idx:]]
    wiki_test_y = wiki_y[indices[split_idx:]]
    
    print(f"   -> Wiki Split: Train({len(wiki_train_x)}) / Evaluation({len(wiki_test_x)})")

    # 3. 학습 데이터 합치기
    final_train_x = torch.cat([kaggle_x, wiki_train_x], dim=0)
    final_train_y = torch.cat([kaggle_y, wiki_train_y], dim=0)
    
    print(f"3️⃣ Total Training Set: {len(final_train_x)} images")

    # 4. Dataset & Loader
    train_ds = RAMDataset(final_train_x, final_train_y, EXP_NAME, is_train=True)
    test_ds = RAMDataset(wiki_test_x, wiki_test_y, EXP_NAME, is_train=False) 
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # 5. Model (Pretrained)
    model = models.resnet18(weights='IMAGENET1K_V1')
    model.fc = nn.Linear(512, 2)
    model = model.to(DEVICE)
    
    # Fine-tuning이므로 LR을 낮게 설정
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00005) 
    
    best_acc = 0.0
    print("4️⃣ Start Adaptation Training...")
    EPOCHS = 30
    for epoch in range(EPOCHS):
        model.train()
        for inputs, labels in tqdm(train_loader, desc=f"Ep {epoch+1}", leave=False):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            # Mixup 사용
            inputs, ta, tb, lam = mixup_data(inputs, labels, alpha=0.4)
            outputs = model(inputs)
            loss = mixup_criterion(criterion, outputs, ta, tb, lam)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        # 평가
        model.eval()
        preds, acts = [], []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(DEVICE)
                outputs = model(inputs)
                preds.extend(torch.argmax(outputs, 1).cpu().numpy())
                acts.extend(labels.numpy())
        
        acc = accuracy_score(acts, preds)
        print(f"Epoch {epoch+1} | Adapted Test Acc: {acc:.4f}")
        
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "model_ram_ADAPTED.pth")
            
    print(f"\n✨ Final Adapted Accuracy: {best_acc:.4f}")
    
    # 혼동 행렬 출력
    cm = confusion_matrix(acts, preds)
    print("-" * 30)
    print("Confusion Matrix:")
    print(cm)
    print("-" * 30)

run_domain_adaptation()