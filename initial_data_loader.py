import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from pie_data import PIE
import torch
from torch.utils.data import DataLoader, TensorDataset

def clear_cache(pie_path):
    cache_files = [
        os.path.join(pie_path, 'data_cache', 'pie_database.pkl'),
        os.path.join(pie_path, 'data_cache', 'random_samples.pkl'),
        os.path.join(pie_path, 'data_cache', '5_fold_samples.pkl')
    ]
    for cache_file in cache_files:
        if os.path.isfile(cache_file):  # Check if file exists before removal
            os.remove(cache_file)
            print(f"Removed cache file: {cache_file}")
        else:
            print(f"Cache file not found: {cache_file}")

def load_full_data(data_opts, pie_path):
    pie = PIE(data_path=pie_path)
    def generate_full_sequence_data():
        print("\nGenerating full data sequence...")
        sequence_data = pie.generate_data_trajectory_sequence('train', **data_opts)
        print("Finished generating training data.")
        return sequence_data
    sequence_data = generate_full_sequence_data()
    return sequence_data

def preprocess_data(data, image_size=(224, 224), num_frames=10, num_channels=3):
    images = []
    labels = []

    for img_seq, label_seq, prob_seq in zip(data['image'], data['intention_binary'], data['intention_prob']):
        current_images = []
        set_vid_pair = None

        for img_path, label, prob in zip(img_seq, label_seq, prob_seq):
            if os.path.isfile(img_path):
                img = cv2.imread(img_path)
                if img is None:  # Catch cases where the image could not be read
                    print(f"Warning: Could not read image {img_path}, skipping.")
                    continue


                # Process the image
                img = cv2.resize(img, image_size)
                if num_channels == 1:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    img = np.expand_dims(img, axis=0)  # Add channel dimension for grayscale
                else:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = img.transpose(2, 0, 1)  # Change from HWC to CHW format

                current_images.append(img)

                if len(current_images) >= num_frames:
                    break  # Stop collecting frames for this sequence

            else:
                print(f"Warning: File {img_path} not found, skipping.")

        # Check for insufficient number of frames in a sequence
        if len(current_images) < num_frames:
            print(f"Warning: Insufficient number of frames ({len(current_images)}) found in sequence, skipping.")
            continue

        images.append(np.array(current_images))
        labels.append(label[0])  # Using the first label in the sequence as the label for the entire sequence

        # Print to validate
        print(f"label[0]: {label[0]}, intention_prob: {prob[0]}")

    images = np.array(images, dtype=np.float32) / 255.0  # Normalize
    labels = np.array(labels, dtype=np.float32)
    
    return images, labels

def split_data(images, labels, test_size=0.2, val_size=0.1):
    # Split into train+val and test
    X_train_val, X_test, y_train_val, y_test = train_test_split(images, labels, test_size=test_size, random_state=42)
    # Split train+val into train and val
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=val_size_adjusted, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test

# Path to PIE dataset root
PIE_PATH = './PIE_dataset'

# Clear cache files
clear_cache(PIE_PATH)

# Define data options
data_opts = {
    'fstride': 1,
    'data_split_type': 'default',
    'seq_type': 'intention',
    'height_rng': [0, float('inf')],
    'squarify_ratio': 0,
    'min_track_size': 0,
    'random_params': {'ratios': None, 'val_data': True, 'regen_data': True},
    'kfold_params': {'num_folds': 5, 'fold': 1}
}

# full_data = load_full_data(data_opts, PIE_PATH)

# Preprocess the data
# images, labels = preprocess_data(full_data)

# Split the data into training, validation, and test sets
# X_train, X_val, X_test, y_train, y_val, y_test = split_data(images, labels)

# Now X_train, X_val, X_test, y_train, y_val, and y_test are ready to be used for training, validation, and testing
