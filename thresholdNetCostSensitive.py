
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
from data_loader_processed import load_full_data, preprocess_data, split_data
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

from data_loader_processed import preprocess_data
from pie_data import PIE
# Load the annotations once
pie_data_instance = PIE(data_path=PIE_PATH)

#create annotations
annotations = pie_data_instance.generate_database()

# Preprocess the data with the pre-loaded annotations
images, labels, set_vid_ids, contexts, frame_ids_list = preprocess_data(full_data, pie_data_instance, annotations)

# Verify the outputs
for i, (img_seq, label, set_vid, context_seq, frame_ids) in enumerate(zip(images, labels, set_vid_ids, contexts, frame_ids_list)):
    print(f"\n--- Sample {i+1} ---")
    print(f"Set ID: {set_vid[0]}, Video ID: {set_vid[1]}")
    print(f"Label: {label}")
    print(f"Frame IDs: {frame_ids}")
    print(f"Context for first frame in this sequence: {context_seq[0]}")  # Print the context of the first frame
    print(f"Number of frames in this sequence: {len(context_seq)}")
    
    if i >= 2:  # Limit the number of samples printed for verification
        break


# %%

#pie_data_instance.extract_and_save_images(extract_frame_type='annotated')


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
from data_loader_processed import dynamic_threshold
from torch.utils.data import DataLoader, TensorDataset
from models import get_resnet50_model, get_vgg16_model, get_alexnet_model
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# %%
import torch

def context_to_tensor(context_lists):
    context_tensors = []
    for context_sequence in context_lists:
        sequence_tensor = []
        for context in context_sequence:
            # Convert each context dictionary into a flat list of values
            flat_context = [
                context['OBD_speed'],
                context['GPS_speed'],
                context['heading_angle'],
                context['ped_action'],
                context['ped_gesture'],
                context['ped_look'],
                context['ped_cross'],
                context['ped_occlusion'],
                context.get('traffic_light_type', 0),  # Use default value 0 if key is missing
                context.get('traffic_light_state', 0),  # Use default value 0 if key is missing
                int(context['crosswalk_present']),  # Convert boolean to int (0 or 1)
                context.get('sign_type', 0)  # Use default value 0 if key is missing
            ]
            sequence_tensor.append(flat_context)
        
        # Convert the sequence of flat contexts into a tensor
        sequence_tensor = torch.tensor(sequence_tensor, dtype=torch.float32)
        context_tensors.append(sequence_tensor)
    
    # Stack context tensors into a single tensor
    context_tensor = torch.stack(context_tensors)
    return context_tensor

contexts_tensor = context_to_tensor(contexts)  # Convert the list of contexts to a tensor



# %%
print("Images shape:", images.shape)
print("Labels shape:", labels.shape)
print("Contexts shape:", contexts_tensor.shape)


# %%


# %%
images = torch.tensor(images).view(-1, 3, 224, 224)
labels = torch.tensor(labels).repeat_interleave(images.shape[0] // labels.shape[0])

# Define transformations
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Apply transformations to each image
images = torch.stack([transform(img) for img in images])

# Reshape images and labels to match context sequences
images = images.view(-1, 10, 3, 224, 224)  # Group images into sequences of 10
labels = labels.view(-1, 10)  # Group labels into sequences of 10

dataset = TensorDataset(images, labels, contexts_tensor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# %%
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from models import get_resnet50_model, get_vgg16_model, get_alexnet_model
import torch.nn.functional as F

class WeightedBCELoss(nn.Module):
    def __init__(self, pos_weight=1.0, neg_weight=1.0):
        super(WeightedBCELoss, self).__init__()
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight

    def forward(self, outputs, targets):
        loss = - (self.pos_weight * targets * torch.log(outputs + 1e-8) +
                  self.neg_weight * (1 - targets) * torch.log(1 - outputs + 1e-8))
        return loss.mean()


# Define the cost-sensitive loss function with higher penalty on false negatives
pos_weight = 2.0  # Higher weight for the positive class (crossing)
neg_weight = 1.0  # Lower weight for the negative class (not crossing)
criterion = WeightedBCELoss(pos_weight=pos_weight, neg_weight=neg_weight)

# List of models to train
models = {
    'resnet50': get_resnet50_model(pretrained=True, num_classes=1),
    'vgg16': get_vgg16_model(pretrained=True, num_classes=1),
    'alexnet': get_alexnet_model(pretrained=True, num_classes=1)
}

num_epochs = 10
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# %%
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from models import get_resnet50_model, get_vgg16_model, get_alexnet_model
import torch.nn.functional as F

def verify_data_and_models(models_dict, dataloader, device='cpu'):
    """
    Verifies that input data, model architectures, and training processes are functioning correctly.
    
    Args:
    - models_dict (dict): Dictionary of model names and model instances.
    - dataloader (DataLoader): DataLoader object for iterating over the dataset.
    - device (str): The device on which to perform the verification ('cpu' or 'cuda').
    
    Returns:
    - None
    """
    # Check the data
    for i, (inputs, targets, context) in enumerate(dataloader):
        print(f"Batch {i+1}:")
        print(f"  Inputs shape: {inputs.shape}")
        print(f"  Targets shape: {targets.shape}")
        print(f"  Context shape: {context.shape}")
        
        # Check if all inputs are unique within a batch
        flattened_inputs = inputs.view(inputs.size(0), -1)
        unique_inputs = torch.unique(flattened_inputs, dim=0)
        if unique_inputs.size(0) != flattened_inputs.size(0):
            print("  Warning: Duplicate inputs found in batch.")
        
        # Check if all targets are correct within a batch
        unique_targets = torch.unique(targets)
        print(f"  Unique targets in batch: {unique_targets.tolist()}")
        
        if i == 0:  # Only check the first batch for now
            break

    # Check model architectures and initialization
    for model_name, model in models_dict.items():
        model = model.to(device)
        print(f"\nVerifying model: {model_name}")
        

        # Print initial parameters (first layer's weights)
        for name, param in model.named_parameters():
            if "weight" in name and param.requires_grad:
                print(f"  Initial {name}: Mean={param.mean().item():.6f}, Std={param.std().item():.6f}")
                break
        
        # Forward pass on a single batch to check output
        inputs, targets, context = next(iter(dataloader))
        inputs, targets, context = inputs.to(device), targets.to(device), context.to(device)
        with torch.no_grad():
            outputs = model(inputs.view(-1, 3, 224, 224))
        print(f"  Outputs shape: {outputs.shape}")
        print(f"  Outputs sample: {outputs[:5].squeeze().tolist()}")
    
    # Check optimizer states
    for model_name, model in models_dict.items():
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        print(f"\nOptimizer state for {model_name}:")
        for i, param_group in enumerate(optimizer.param_groups):
            print(f"  Param group {i}: Learning rate = {param_group['lr']}")
            for param in param_group['params']:
                if param.requires_grad:
                    print(f"    Param {param.shape}: Mean={param.data.mean().item():.6f}, Std={param.data.std().item():.6f}")
                    break

    # Check random seed settings
    print("\nRandom seed verification:")
    seed = torch.initial_seed()
    print(f"  Global random seed: {seed}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            cuda_seed = torch.cuda.initial_seed()
            print(f"  CUDA device {i} seed: {cuda_seed}")



# List of models to train
models = {
    'resnet50': get_resnet50_model(pretrained=True, num_classes=1),
    'vgg16': get_vgg16_model(pretrained=True, num_classes=1),
    'alexnet': get_alexnet_model(pretrained=True, num_classes=1)
}

# Verify the data and models
verify_data_and_models(models, dataloader)


# %% [markdown]
# ### Training WITH COST-SENSITIVE LEARNING

# %%
for model_name, model in models.items():
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(f"Training {model_name} with cost-sensitive learning...")

    for epoch in range(num_epochs):
        model.train()
        all_targets, all_predictions = [], []

        for inputs, targets, context in dataloader:
            inputs, targets, context = inputs.to(device), targets.to(device), context.to(device)

            # Flatten the inputs to process through the model
            inputs = inputs.view(-1, 3, 224, 224)
            targets = targets.view(-1)  # Flatten targets to match inputs

            outputs = model(inputs)
            outputs = torch.sigmoid(outputs).squeeze()

            # Calculate the weighted loss
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            predicted = (outputs > 0.5).float()

            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

        # Calculate metrics
        accuracy = accuracy_score(all_targets, all_predictions)
        precision = precision_score(all_targets, all_predictions)
        recall = recall_score(all_targets, all_predictions)
        f1 = f1_score(all_targets, all_predictions)

        print(f"Epoch {epoch+1}/{num_epochs}: Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

        # Save the model after cost-sensitive learning
        model_path = f'./saved_models/{model_name}_intention_binary_cost_sensitive.pth'
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")


# Final evaluation on the test set
for model_name, model in models.items():
    model = model.to(device)
    model.eval()
    all_targets, all_predictions = [], []

    with torch.no_grad():
        for inputs, targets, context in dataloader:
            inputs, targets, context = inputs.to(device), targets.to(device), context.to(device)

            # Flatten the inputs to process through the model
            inputs = inputs.view(-1, 3, 224, 224)
            targets = targets.view(-1)  # Flatten targets to match inputs

            outputs = model(inputs)
            outputs = torch.sigmoid(outputs).squeeze()

            predicted = (outputs > 0.5).float()

            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(all_targets, all_predictions)
    precision = precision_score(all_targets, all_predictions)
    recall = recall_score(all_targets, all_predictions)
    f1 = f1_score(all_targets, all_predictions)

    print(f"{model_name} Test Performance:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")

# %% [markdown]
# ### Training WITH THRESHOLD NETWORK 

# %%
class ThresholdNet(nn.Module):
    def __init__(self, input_dim, output_dim=1):
        super(ThresholdNet, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        x = torch.sigmoid(self.fc(x))
        return x


#Instantiate the threshold network
threshold_net = ThresholdNet(input_dim=12).to(device)



# Combine the threshold network with the main model training loop
for model_name, model in models.items():
    model = model.to(device)
    
    
    # Load the saved model state from the cost-sensitive learning phase
    model_path = f'./saved_models/{model_name}_intention_binary_cost_sensitive.pth'
    model.load_state_dict(torch.load(model_path))
    print(f"Model {model_name} loaded from {model_path}")

    criterion = nn.BCELoss()
    optimizer = optim.Adam(list(model.parameters()) + list(threshold_net.parameters()), lr=0.001)

    print(f"Training {model_name}...")

    for epoch in range(num_epochs):
        model.train()
        all_targets, all_predictions = [], []

        for inputs, targets, context in dataloader:
            inputs, targets, context = inputs.to(device), targets.to(device), context.to(device)

            # Flatten the inputs to process through the model
            inputs = inputs.view(-1, 3, 224, 224)
            targets = targets.view(-1)  # Flatten targets to match inputs

            outputs = model(inputs)
            outputs = torch.sigmoid(outputs).squeeze()

            # Generate dynamic thresholds
            thresholds = threshold_net(context.view(-1, 12)).squeeze()

            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            predicted = (outputs > thresholds).float()

            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

        # Calculate metrics
        accuracy = accuracy_score(all_targets, all_predictions)
        precision = precision_score(all_targets, all_predictions)
        recall = recall_score(all_targets, all_predictions)
        f1 = f1_score(all_targets, all_predictions)

        print(f"Epoch {epoch+1}/{num_epochs}: Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

    # Save the model
    model_path = f'./saved_models/{model_name}_intention_binary_threshold_net.pth'
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

# Final evaluation on the test set
for model_name, model in models.items():
    model = model.to(device)
    model.eval()

    # Load the model state for evaluation
    model_path = f'./saved_models/{model_name}_intention_binary_threshold_net.pth'
    model.load_state_dict(torch.load(model_path))
    print(f"Model {model_name} loaded from {model_path} for final evaluation.")


    # Initialize the ThresholdNet if using dynamic thresholds
    threshold_net = ThresholdNet(input_dim=12).to(device)
    threshold_net.eval()  # Set to evaluation mode

    all_targets, all_predictions = [], []

    with torch.no_grad():
        for inputs, targets, context in dataloader:
            inputs, targets, context = inputs.to(device), targets.to(device), context.to(device)

            # Flatten the inputs to process through the model
            inputs = inputs.view(-1, 3, 224, 224)
            targets = targets.view(-1)  # Flatten targets to match inputs

            outputs = model(inputs)
            outputs = torch.sigmoid(outputs).squeeze()

            # Generate dynamic thresholds using the ThresholdNet
            thresholds = threshold_net(context.view(-1, 12)).squeeze()

            # Apply the predicted thresholds
            predicted = (outputs > thresholds).float()

            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(all_targets, all_predictions)
    precision = precision_score(all_targets, all_predictions)
    recall = recall_score(all_targets, all_predictions)
    f1 = f1_score(all_targets, all_predictions)

    print(f"{model_name} Test Performance:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")



# %%
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

def verify_data_and_models(models_dict, dataloader, device='cpu'):
    """
    Verifies that input data, model architectures, and training processes are functioning correctly.
    
    Args:
    - models_dict (dict): Dictionary of model names and model instances.
    - dataloader (DataLoader): DataLoader object for iterating over the dataset.
    - device (str): The device on which to perform the verification ('cpu' or 'cuda').
    
    Returns:
    - None
    """
    # Check the data
    for i, (inputs, targets, context) in enumerate(dataloader):
        print(f"Batch {i+1}:")
        print(f"  Inputs shape: {inputs.shape}")
        print(f"  Targets shape: {targets.shape}")
        print(f"  Context shape: {context.shape}")
        
        # Check if all inputs are unique within a batch
        flattened_inputs = inputs.view(inputs.size(0), -1)
        unique_inputs = torch.unique(flattened_inputs, dim=0)
        if unique_inputs.size(0) != flattened_inputs.size(0):
            print("  Warning: Duplicate inputs found in batch.")
        
        # Check if all targets are correct within a batch
        unique_targets = torch.unique(targets)
        print(f"  Unique targets in batch: {unique_targets.tolist()}")
        
        if i == 0:  # Only check the first batch for now
            break

    # Check model architectures and initialization
    for model_name, model in models_dict.items():
        model = model.to(device)
        print(f"\nVerifying model: {model_name}")
        
        # Print model architecture
        print(f"  Model architecture:\n{model}")
        
        # Print initial parameters (first layer's weights)
        for name, param in model.named_parameters():
            if "weight" in name and param.requires_grad:
                print(f"  Initial {name}: Mean={param.mean().item():.6f}, Std={param.std().item():.6f}")
                break
        
        # Forward pass on a single batch to check output
        inputs, targets, context = next(iter(dataloader))
        inputs, targets, context = inputs.to(device), targets.to(device), context.to(device)
        with torch.no_grad():
            outputs = model(inputs.view(-1, 3, 224, 224))
        print(f"  Outputs shape: {outputs.shape}")
        print(f"  Outputs sample: {outputs[:5].squeeze().tolist()}")
    
    # Check optimizer states
    for model_name, model in models_dict.items():
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        print(f"\nOptimizer state for {model_name}:")
        for i, param_group in enumerate(optimizer.param_groups):
            print(f"  Param group {i}: Learning rate = {param_group['lr']}")
            for param in param_group['params']:
                if param.requires_grad:
                    print(f"    Param {param.shape}: Mean={param.data.mean().item():.6f}, Std={param.data.std().item():.6f}")
                    break

    # Check random seed settings
    print("\nRandom seed verification:")
    seed = torch.initial_seed()
    print(f"  Global random seed: {seed}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            cuda_seed = torch.cuda.initial_seed()
            print(f"  CUDA device {i} seed: {cuda_seed}")

    # Final check to ensure models have independent states
    print("\nFinal check: Models' first layer parameters should differ")
    first_model_name, first_model = list(models_dict.items())[0]
    for model_name, model in models_dict.items():
        if model_name != first_model_name:
            first_layer_name = list(first_model.named_parameters())[0][0]
            first_layer_params_first_model = list(first_model.named_parameters())[0][1].data
            first_layer_params_other_model = list(model.named_parameters())[0][1].data
            difference = torch.sum(torch.abs(first_layer_params_first_model - first_layer_params_other_model))
            if difference == 0:
                print(f"  Warning: First layer parameters of {first_model_name} and {model_name} are identical.")
            else:
                print(f"  Difference in first layer parameters between {first_model_name} and {model_name}: {difference.item()}")


verify_data_and_models(models, dataloader)


