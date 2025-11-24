파일 경로

```/cnn/fourier_transform/```

푸리에 변환 함수명

```def preprocess_FT(img: Image.Image, Type = None):```

googlenet 모델 사용


ft_type = 0 이면 phase_only

ft_type = 1 이면 amplitude_only

```train_dataset = DeepfakeDataset(train_paths, train_labels, transform=train_transform, apply_ft=True, ft_type=1)```

데이터셋: https://www.kaggle.com/datasets/malekbennabi/realfake-cropped-faces
