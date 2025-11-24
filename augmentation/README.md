파일명: ```cnn/deepfake_baseline_pytorch_colab_googleNet.ipynb```

데이터셋: Google Drive에 업로드한 real_fake_dataset을 사용하여 68,479장의 이미지 중 10,000장을 균등 샘플링하여 학습·검증·테스트 셋으로 분할함.
https://www.kaggle.com/datasets/malekbennabi/realfake-cropped-faces

사용 모델: GoogleNet

데이터 증강: Train set에만 augmentation을 적용함
```python
# 전처리 및 데이터 로더 정의
train_transform= transforms.Compose([
    ResizeWithPad(IMG_SIZE),
		transforms.RandomHorizontalFlip(p=0.5),# 좌우 반전
		transforms.RandomRotation(10),#+- 10도 회전
		transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.02
    ),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])# -1 ~ 1 정규화])
```
- 224 X 224 크기의 이미지로 전처리
- 50%의 확률로 이미지를 좌우 반전
- 이미지도 +- 10도 회전
- 조명 및 색감 변화에 대한 적응력 상승 등

Test Accuracy: 79.30%
