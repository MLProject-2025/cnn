# 🎓 DeepFake Detection - CNN Models

A comprehensive deep learning project for detecting deepfake images using various CNN architectures including AlexNet, VGG16, GoogLeNet, ResNet, and Vision Transformer (ViT).

## 📁 Directory Structure

```
cnn/
├── data/                       # Dataset samples, metadata, and test records
│   └── test_records/          # Historical test results and records
├── augmentation/              # Data augmentation scripts (currently in experiments/)
├── finetuning/                # Fine-tuning code and pretrained model logic (currently in experiments/)
├── models/                    # Model definitions (to be organized)
├── training/                  # Training loops, dataloaders, loss, optimizer (to be organized)
├── evaluation/                # Test scripts, metrics, result logging (to be organized)
├── utils/                     # Preprocessing utilities
│   ├── data_check.py         # Dataset analysis and statistics
│   ├── data_loader.py        # Data loading utilities
│   ├── fourier_transform.py  # Fourier Transform preprocessing
│   └── resize_with_pad.py    # Image resizing with padding
├── experiments/               # Jupyter notebooks for experiments and reproducibility
│   ├── augmentation/         # Augmentation experiment notebooks
│   ├── finetuning/           # Fine-tuning experiment notebooks
│   │   └── fft/             # FFT-based fine-tuning experiments
│   ├── fourier_transform/    # Fourier Transform experiments
│   └── test/                 # Test/evaluation notebooks
├── README.md
└── requirements.txt
```

## 🚀 How to Run This Project

### 1. Installation

#### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (recommended for training)
- 8GB+ RAM

#### Install Dependencies

```bash
# Clone the repository
git clone https://github.com/MLProject-2025/cnn.git
cd cnn

# Install required packages
pip install -r requirements.txt
```

### 2. Dataset Preparation

This project uses the **Real/Fake Face Dataset** from Kaggle:
- Dataset URL: https://www.kaggle.com/datasets/malekbennabi/realfake-cropped-faces
- Total images: ~68,000 images
- Classes: Real faces vs. Fake (AI-generated) faces

#### Download and Setup

1. Download the dataset from Kaggle
2. Extract to a directory (e.g., `./dataset/univ_ML_basic/deepfake/original`)
3. The dataset should contain subdirectories for real and fake images

#### Dataset Preprocessing

Check dataset statistics and distribution:
```bash
python utils/data_check.py
```

Create a preprocessed dataset with consistent sizing:
```bash
python utils/resize_with_pad.py
```

This script will:
- Resize images to 224x224 pixels
- Apply padding to maintain aspect ratio
- Sample approximately 10,000 images per class for balanced training

### 3. Training Models

#### Basic Training

The training notebooks are located in `experiments/`. You can run them directly:

**Training with GoogLeNet:**
```bash
jupyter notebook experiments/deepfake_baseline_pytorch_local.ipynb
```

**Training with Data Augmentation:**
```bash
jupyter notebook experiments/augmentation/deepfake_baseline_pytorch_colab_googleNet.ipynb
```

This applies augmentation techniques including:
- Random horizontal flip (50% probability)
- Random rotation (±10 degrees)
- Color jitter (brightness, contrast, saturation, hue)
- ResizeWithPad for 224x224 images
- Normalization to [-1, 1] range

**Expected Results:**
- Test Accuracy: ~79.30% with GoogLeNet + augmentation

### 4. Fine-tuning Pretrained Models

Fine-tuning experiments use transfer learning with pretrained models:

#### GoogLeNet Fine-tuning:
```bash
jupyter notebook experiments/finetuning/train_googlenet_finetuning.ipynb
# OR
jupyter notebook experiments/train_googlenet_finetuning.ipynb
```

#### ResNet-50 Fine-tuning:
```bash
jupyter notebook experiments/train_resnet50_finetuning.ipynb
```

#### Vision Transformer (ViT) Fine-tuning:
```bash
jupyter notebook experiments/finetuning/train_vit_finetuning.ipynb
```

**Fine-tuning Strategy:**
- Load pretrained weights from ImageNet
- Freeze early layers (feature extractors)
- Train final classification layers
- Gradually unfreeze layers for better adaptation

### 5. Fourier Transform Preprocessing

The project includes experiments with Fourier Transform-based preprocessing to analyze frequency domain features:

#### Understanding Fourier Transform Modes

The `utils/fourier_transform.py` provides two main preprocessing modes:

1. **Phase-only** (`ft_type=0`): Retains only phase information, sets amplitude to 1
2. **Amplitude-only** (`ft_type=1`): Removes phase, keeps only amplitude information

#### Running FT Experiments

**Amplitude-only preprocessing:**
```bash
jupyter notebook experiments/fourier_transform/train_googlenet_amplitude_only_ft.ipynb
```

**Phase-only preprocessing:**
```bash
jupyter notebook experiments/fourier_transform/train_googlenet_phase_only_ft.ipynb
```

**FFT Fine-tuning with ResNet:**
```bash
jupyter notebook experiments/finetuning/fft/fft_resnet50_finetuning.ipynb
```

**FFT Fine-tuning with GoogLeNet:**
```bash
jupyter notebook experiments/finetuning/fft/fft_googlenet_finetuning-2.ipynb
```

**FFT Fine-tuning with ViT:**
```bash
jupyter notebook experiments/finetuning/fft/fft_vit_finetuning.ipynb
```

#### FT Transform Demo

Explore Fourier Transform visualization:
```bash
jupyter notebook experiments/fourier_transform/ft_transform.ipynb
```

### 6. Model Evaluation

#### Testing Different Architectures

The `experiments/test/` directory contains evaluation notebooks for various models:

**AlexNet Evaluation:**
```bash
jupyter notebook experiments/test/deepfake_alexnet_dataset.ipynb
jupyter notebook experiments/test/deepfake_alexnet_update_dataset1.ipynb
```

**GoogLeNet Evaluation:**
```bash
jupyter notebook experiments/test/deepfake_goolenet_dataset1.ipynb
jupyter notebook experiments/test/deepfake_goolenet_dataset2.ipynb
```

**ResNet Evaluation:**
```bash
jupyter notebook experiments/test/deepfake_resnet_dataset1.ipynb
jupyter notebook experiments/test/deepfake_resnet_dataset2.ipynb
jupyter notebook experiments/test/deepfake_resnet_keras.ipynb
```

### 7. Advanced Analysis

#### GradCAM Visualization

Visualize which regions of an image the model focuses on:
```bash
jupyter notebook experiments/deepfake_with_gradcam.ipynb
```

GradCAM (Gradient-weighted Class Activation Mapping) helps understand:
- Which facial features the model uses for classification
- Whether the model focuses on artifacts typical of deepfakes
- Model interpretability and debugging

#### Dataset Generalization Study

Evaluate model performance across different datasets:
```bash
jupyter notebook experiments/dataset_generalization_experiment.ipynb
```

## 📊 Model Architectures

The project experiments with multiple CNN architectures:

1. **AlexNet**: Classic deep CNN, 8 layers
2. **VGG16**: Deep architecture with small 3x3 filters
3. **GoogLeNet (Inception v1)**: Multi-scale feature extraction with inception modules
4. **ResNet-50**: Residual connections for very deep networks
5. **Vision Transformer (ViT)**: Attention-based transformer architecture

## 🔧 Utilities

### ResizeWithPad
Maintains aspect ratio while resizing images to target size with black padding.

```python
from utils.resize_with_pad import resize_with_pad
resized_img = resize_with_pad(image, target_size=224)
```

### Fourier Transform Preprocessing
Apply frequency domain transformations for feature analysis.

```python
from utils.fourier_transform import preprocess_FT
# Phase-only
transformed_img = preprocess_FT(image, Type=0)
# Amplitude-only
transformed_img = preprocess_FT(image, Type=1)
```

### Data Checking
Analyze dataset statistics, file counts, and image dimensions.

```python
python utils/data_check.py
```

## 📝 Experiment Notes

- **Augmentation Notes**: See `experiments/augmentation_notes.md` for detailed augmentation strategies
- **Fourier Transform Notes**: See `experiments/fourier_transform_notes.md` for FT implementation details
- **Test Records**: Historical test results are stored in `data/test_records/`

## 🎯 Quick Start Example

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download and prepare dataset
python utils/data_check.py  # Check your dataset

# 3. Run a training experiment
jupyter notebook experiments/deepfake_baseline_pytorch_local.ipynb

# 4. Evaluate with GradCAM
jupyter notebook experiments/deepfake_with_gradcam.ipynb
```

## 📚 References

- Dataset: [Real/Fake Face Dataset](https://www.kaggle.com/datasets/malekbennabi/realfake-cropped-faces)
- Framework: PyTorch
- Pretrained Models: torchvision.models

## 👤 Contributors

| Contributor | Contributor | Contributor | Contributor |
|------------|-------------|-------------|-------------|
| <img src="https://avatars.githubusercontent.com/woals2840" width="100px;"><br/>**곽재민** | <img src="https://avatars.githubusercontent.com/u/169603063?v=4" width="100px"><br/>**박동욱** |<img src="https://avatars.githubusercontent.com/u/162528062?v=4" width="100px"><br/>**이예영** | <img src="https://avatars.githubusercontent.com/u/161849407?v=4" width="100px"><br/>**이채윤** |

## 📄 License

This project is for educational purposes as part of the Machine Learning course.

## 🤝 Contributing

This is an academic project. For questions or suggestions, please open an issue.
