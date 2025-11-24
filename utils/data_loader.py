"""
Data loading utilities for deepfake detection.

This module provides functions to load and sample real/fake image datasets.
"""

import os
import glob
import numpy as np
from sklearn.model_selection import train_test_split


def load_and_sample(real_dir, fake_dir, n_samples=None):
    real_paths = []
    fake_paths = []

    real_paths.extend(glob.glob(os.path.join(real_dir, "*.*")))
    fake_paths.extend(glob.glob(os.path.join(fake_dir, "*.*")))

    paths = real_paths + fake_paths
    labels = [0] * len(real_paths) + [1] * len(fake_paths)
    total = len(paths)

    print(f"Path: {os.path.dirname(real_dir)}")
    print(f"Count: {total} images (Real: {len(real_paths)}, Fake: {len(fake_paths)})")

    if n_samples is None or n_samples >= total:
        print(f"  - Using all data ({total} images)")
        # Shuffle only
        combined = list(zip(paths, labels))
        np.random.shuffle(combined)
        paths[:], labels[:] = zip(*combined)
        return paths, labels
    
    else:
        print(f"  - Sampling: {n_samples} images selected (Stratified)")

        sampled_paths, _, sampled_labels, _ = train_test_split(
            paths, labels,
            train_size=n_samples,
            stratify=labels,
            random_state=42
        )
        return sampled_paths, sampled_labels


def main():
    """Example usage of the data loader."""
    # Data paths and label collection
    base_path = "../datasets/deepfake/"
    train_path = base_path + 'Train'
    validdation_path = base_path + 'Validation'
    test_path = base_path + 'Test'
    
    NUM_TRAIN_SAMPLES = 50000 
    NUM_VAL_SAMPLES = 5000     
    NUM_TEST_SAMPLES = None    # None: use all data
    
    print("\n=== [Train Set] Preparation ===")
    train_paths, train_labels = load_and_sample(
        os.path.join(train_path, 'Real'), 
        os.path.join(train_path, 'Fake'), 
        NUM_TRAIN_SAMPLES
    )
    
    print("\n=== [Validation Set] Preparation ===")
    val_paths, val_labels = load_and_sample(
        os.path.join(validdation_path, 'Real'), 
        os.path.join(validdation_path, 'Fake'), 
        NUM_VAL_SAMPLES
    )
    
    print("\n=== [Test Set] Preparation ===")
    test_paths, test_labels = load_and_sample(
        os.path.join(test_path, 'Real'), 
        os.path.join(test_path, 'Fake'), 
        NUM_TEST_SAMPLES
    )
    
    # 4. Check results
    print(f"\nFinal Dataset Configuration:")
    print(f"  Train: {len(train_paths)} images")
    print(f"  Val  : {len(val_paths)} images")
    print(f"  Test : {len(test_paths)} images")


if __name__ == "__main__":
    main()
