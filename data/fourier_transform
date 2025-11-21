def preprocess(path, FT = None):  # 이미지 전처리
  img = Image.open(path)

  if FT != None:
    img = pre_transform(img)  # 전전처리 적용
    img = preprocess_FT(img, FT)
    processed_img_tensor = after_FT_transform(img)  # 후전처리 적용

  elif FT == None:
    processed_img_tensor = transform(img)
  
  return processed_img_tensor


FT_type = ['phase_only', 'amplitude_only', 'shuffle_global_amplitude', 'shuffle_patch_amplitude', 'average_amplitude', 'shuffle_patch_phase']

# 푸리에 변환 함수
def preprocess_FT(img, Type = None):
  Type = FT_type[Type]

  img_np = np.array(img)

  ifft_channels = []
  for i in range(img_np.shape[2]): # 채널 반복
    fft_result = np.fft.fft2(img_np[:, :, i])

    # --------- 다양한 FT 처리 방식 ---------
    if Type == 'phase_only': # 위상 정보만 남김
      left_frequency = np.exp(1j * np.angle(fft_result)) 
      ifft_result = np.fft.ifft2(left_frequency)
      ifft_result = np.real(ifft_result)

    elif Type == 'amplitude_only': # 진폭 정보만 남김
      # left_frequency = np.abs(fft_result)
      # ifft_result = left_frequency
      fft_shifted = np.fft.fftshift(fft_result)        
      amplitude_spectrum = np.log(np.abs(fft_shifted) + 1e-8)
      ifft_result = amplitude_spectrum

    def scaling(img_np):
      # 스케일링 후 PIL 이미지로 변환
      min_val, max_val = np.min(img_np), np.max(img_np)
      scaled_np = 255.0 * (img_np - min_val) / (max_val - min_val + 1e-9)
      return Image.fromarray(scaled_np.astype(np.uint8))
    
    ifft_result = scaling(ifft_result)
    ifft_channels.append(ifft_result)

  ifft_image = np.stack(ifft_channels, axis=-1)

  return ifft_image
