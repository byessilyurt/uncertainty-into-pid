
# Set environment variables
import os
import sys

# Path to PIE dataset root
PIE_PATH = './PIE_dataset'
# Path to PIE raw data (replace with actual path if different)
PIE_RAW_PATH = './PIE_dataset/PIE_clips'

# Set environment variables
os.environ['PIE_PATH'] = PIE_PATH
os.environ['PIE_RAW_PATH'] = PIE_RAW_PATH

# Verify environment variables
print('PIE_PATH:', os.environ['PIE_PATH'])
print('PIE_RAW_PATH:', os.environ['PIE_RAW_PATH'])

# Add current directory to Python path
sys.path.append('./')

# Create necessary directories
MODEL_SAVE_PATH = './saved_models'
if not os.path.exists(MODEL_SAVE_PATH):
    os.makedirs(MODEL_SAVE_PATH)

# Import necessary libraries
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from data_loader import load_full_data, preprocess_data, split_data
from pie_data import PIE

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
PIE_PATH = "./PIE_dataset"

# Load full data
full_data = load_full_data(data_opts, PIE_PATH)


# %%
# pie_data_instance = PIE(data_path=PIE_PATH)
# pie_data_instance.extract_and_save_images(extract_frame_type='annotated')

# %%

from data_loader import preprocess_data
# Preprocess the data
images, labels = preprocess_data(full_data)


# %%
# Print out the shape of the processed data
print(f"Images shape: {images.shape}")
print(f"Labels shape: {labels.shape}")

# Print out the first 5 labels to see the distribution
print("First 5 labels:", labels[:5])


import matplotlib.pyplot as plt
from pie_data import PIE
imdb = PIE(data_path=PIE_PATH)
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def display_image_and_features(images, labels, full_data, imdb, set_id, vid_id, index=0, frame=0):
    # Display the image
    image = images[index][frame]
    if image.shape[0] == 3:  # RGB
        plt.imshow(image.transpose(1, 2, 0))  # Convert CHW to HWC
    else:  # Grayscale
        plt.imshow(image[0], cmap='gray')
    plt.title(f"Label: {labels[index]}")
    plt.axis('off')
    
    # Get the pedestrian ID and retrieve features
    pedestrian_id = full_data['ped_id'][index][frame][0]  # Get the pedestrian ID from the data
    annotations = imdb._get_annotations(set_id, vid_id)
    ped_attributes = imdb._get_ped_attributes(set_id, vid_id)
    
    if pedestrian_id in annotations['ped_annotations']:
        pedestrian_features = annotations['ped_annotations'][pedestrian_id]
        ped_attr = ped_attributes.get(pedestrian_id, {})

        # Get bounding box coordinates
        bbox = pedestrian_features['bbox'][frame]
        x1, y1, x2, y2 = bbox

        # Create a Rectangle patch
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='r', facecolor='none')
        
        # Add the patch to the Axes
        plt.gca().add_patch(rect)

        print(f"Features for Image {index}, Frame {frame}, Pedestrian ID: {pedestrian_id}:")
        print("Bounding Box:", bbox)
        print("Action:", pedestrian_features['behavior']['action'][frame])
        print("Gesture:", pedestrian_features['behavior']['gesture'][frame])
        print("Looking:", pedestrian_features['behavior']['look'][frame])
        print("Crossing:", pedestrian_features['behavior']['cross'][frame])
        print("Occlusion:", pedestrian_features['occlusion'][frame])
        print("---- Object Attributes ----")
        print("Age:", ped_attr.get('age', 'Unknown'))
        print("Gender:", ped_attr.get('gender', 'Unknown'))
        print("Signalized:", ped_attr.get('signalized', 'Unknown'))
        print("Traffic Direction:", ped_attr.get('traffic_direction', 'Unknown'))
        print("Intersection:", ped_attr.get('intersection', 'Unknown'))
        print("Crossing:", ped_attr.get('crossing', 'Unknown'))
        print("Number of Lanes:", ped_attr.get('num_lanes', 'Unknown'))
        print("Intention Probability:", ped_attr.get('intention_prob', 'Unknown'))
        print("Crossing Point:", ped_attr.get('crossing_point', 'Unknown'))
    else:
        print(f"Pedestrian ID {pedestrian_id} not found in annotations.")
    
    # Show the image with bounding box
    plt.show()
    
# Display the first frame of the first image in the dataset
set_id = 'set01'  # Replace with the appropriate set ID
vid_id = 'video_0001'  # Replace with the appropriate video ID
for i in range(3):  # Adjust range for the number of samples you want to see
    display_image_and_features(images, labels, full_data, imdb, set_id=set_id, vid_id=vid_id, index=i, frame=0)


# %%
import ssl
import certifi
ssl._create_default_https_context = ssl._create_unverified_context


# %%
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
from models import get_resnet50_model, get_vgg16_model, get_alexnet_model

# Assuming `images` and `labels` are already processed and loaded
images = torch.tensor(images).view(-1, 3, 224, 224)
labels = torch.tensor(labels).repeat_interleave(images.shape[0] // labels.shape[0])

# Define transformations
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Apply transformations
images = torch.stack([transform(img) for img in images])

# Create TensorDataset and DataLoader
dataset = TensorDataset(images, labels)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# List of models to train
models = {
    'resnet50': get_resnet50_model(pretrained=True, num_classes=1),
    'vgg16': get_vgg16_model(pretrained=True, num_classes=1),
    'alexnet': get_alexnet_model(pretrained=True, num_classes=1)
}

num_epochs = 10
device = 'cuda' if torch.cuda.is_available() else 'cpu'

for model_name, model in models.items():
    model = model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(f"Training {model_name}...")

    for epoch in range(num_epochs):
        model.train()
        total_preds = total_correct = total_fn = total_fp = 0

        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            outputs = torch.sigmoid(outputs).squeeze()

            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Calculate metrics
            predicted = (outputs > 0.5).float()
            total_preds += targets.size(0)
            total_correct += (predicted == targets).sum().item()
            total_fn += ((predicted == 0) & (targets == 1)).sum().item()
            total_fp += ((predicted == 1) & (targets == 0)).sum().item()

        # Calculate rates
        accuracy = 100 * total_correct / total_preds
        fnr = 100 * total_fn / (total_fn + (targets == 1).sum().item())
        fpr = 100 * total_fp / (total_fp + (targets == 0).sum().item())

        print(f"Epoch {epoch+1}/{num_epochs}: Accuracy: {accuracy:.2f}%, FNR: {fnr:.2f}%, FPR: {fpr:.2f}%")

    # Save the model
    model_path = f'./saved_models/{model_name}_intention_binary.pth'
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")



# %%
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Evaluate each model
for model_name, model in models.items():
    model = model.to(device)
    model.eval()
    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            outputs = torch.sigmoid(outputs).squeeze()
            predicted = (outputs > 0.5).float()

            # Collect all targets and predictions to compute global metrics
            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    # Compute metrics
    precision = precision_score(all_targets, all_predictions)
    recall = recall_score(all_targets, all_predictions)
    f1 = f1_score(all_targets, all_predictions)
    accuracy = accuracy_score(all_targets, all_predictions)

    # Print results
    print(f"{model_name} Performance:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  F1 Score: {f1:.4f}")



# %%
%run testi_processed.ipynb
#


