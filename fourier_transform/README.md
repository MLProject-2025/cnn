파일 경로: ```/cnn/fourier_transform/```

사용 모델: ```GoogleNet```

푸리에 변환 함수

```python
def preprocess_FT(img: Image.Image, Type = None):
  Type = FT_type[Type]

  img_np = np.array(img).astype(np.float32)

  H, W, C = img_np.shape
  out_channels = []

  for i in range(C): # 채널 반복
    channel = img_np[:, :, i]

    fft = np.fft.fft2(channel)
    amp = np.abs(fft)
    phase = np.angle(fft)

    # --------- 다양한 FT 처리 방식 ---------
    if Type == 'phase_only': # 위상 정보만 남김
      # amplitude = 1, phase만 유지
      new_fft = np.exp(1j * phase)

    elif Type == 'amplitude_only': # 진폭 정보만 남김
      # left_frequency = np.abs(fft_result)
      # ifft_result = left_frequency
      new_fft = amp

    else:
      # 아직 구현 안 한 타입들은 기본 원본 유지
      new_fft = fft

    if Type in ["phase_only"]:
      ifft = np.fft.ifft2(new_fft)
    elif Type == "amplitude_only":
      ifft = np.log(amp + 1e-8)
    else:
      ifft = np.fft.ifft(new_fft)

    # 실수부만
    ifft = np.real(ifft)
    # 채널 저장
    out_channels.append(ifft)

  out_img = np.stack(out_channels, axis=-1)

  # 0~255 스케일링
  out_img = out_img - out_img.min()
  out_img = out_img / (out_img.max() + 1e-8)
  out_img = (out_img * 255).astype(np.uint8)

  return Image.fromarray(out_img)
```
- phase_only: amplitude는 1로 고정, phase 정보만 유지
- amplitude_only: phase 정보 제거, amplitude만 남긴 상태로 복원


```train_dataset = DeepfakeDataset(train_paths, train_labels, transform=train_transform, apply_ft=True, ft_type=1)```

ft_type = 0 이면 phase_only

ft_type = 1 이면 amplitude_only

데이터셋: https://www.kaggle.com/datasets/malekbennabi/realfake-cropped-faces
