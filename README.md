# CNN Deepfake Detection Project

## Project Structure

```
├── notebooks/              # Jupyter notebooks organized by category
│   ├── baseline/          # Baseline model experiments
│   │   ├── deepfake_baseline_pytorch_colab_googleNet.ipynb
│   │   └── deepfake_baseline_pytorch_local.ipynb
│   ├── finetuning/        # Model finetuning experiments
│   │   ├── train_googlenet_finetuning.ipynb
│   │   ├── train_resnet50_finetuning.ipynb
│   │   ├── train_vit_finetuning.ipynb
│   │   └── train_vit_finetuning-2.ipynb
│   ├── fourier_transform/ # Fourier transform-based experiments
│   │   ├── ft_transform.ipynb
│   │   ├── train_googlenet_amplitude_only_ft.ipynb
│   │   └── train_googlenet_phase_only_ft.ipynb
│   ├── experiments/       # Advanced experiments
│   │   ├── deepfake_with_gradcam.ipynb
│   │   └── domain_generalization_experiment.ipynb
│   └── archive/           # Old test notebooks and results
├── scripts/               # Python utility scripts
│   ├── data_check.py
│   ├── fourier_transform.py
│   ├── new_data_paths_load.py
│   └── sampling_and_resizing.py
└── README.md
```

## Getting Started

1. Upload notebook files to Google Drive
2. Upload and extract data files in the same location
3. Update data paths in the notebooks accordingly
4. Run the notebooks

## Notes

For questions or issues, please contact the project team.
