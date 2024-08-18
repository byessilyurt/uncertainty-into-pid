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

def get_context_for_frame(annotations, set_id, vid_id, frame_id):
    """
    Retrieves contextual information for a given frame.

    Args:
    - annotations: The pre-loaded annotations database.
    - set_id: The set identifier (e.g., 'set01').
    - vid_id: The video identifier (e.g., 'video_0001').
    - frame_id: The specific frame number to retrieve context for.

    Returns:
    - context: A dictionary containing relevant contextual information.
    """
    context = {}

    # Retrieve annotations for the specific video
    video_annotations = annotations[set_id][vid_id]

    # Retrieve vehicle-related information for the specific frame
    vehicle_annotations = video_annotations['vehicle_annotations'].get(frame_id, {})
    context['OBD_speed'] = vehicle_annotations.get('OBD_speed', None)
    context['GPS_speed'] = vehicle_annotations.get('GPS_speed', None)
    context['heading_angle'] = vehicle_annotations.get('heading_angle', None)

    # Retrieve pedestrian-related information
    pedestrian_annotations = video_annotations['ped_annotations']
    for ped_id, ped_data in pedestrian_annotations.items():
        if frame_id in ped_data['frames']:
            index = ped_data['frames'].index(frame_id)
            context['ped_action'] = ped_data['behavior']['action'][index]
            context['ped_gesture'] = ped_data['behavior']['gesture'][index]
            context['ped_look'] = ped_data['behavior']['look'][index]
            context['ped_cross'] = ped_data['behavior']['cross'][index]
            context['ped_occlusion'] = ped_data['occlusion'][index]
            break  # Assuming one pedestrian per frame for simplicity, adjust as needed

    # Retrieve traffic light-related information
    traffic_annotations = video_annotations['traffic_annotations']
    for obj_id, obj_data in traffic_annotations.items():
        if obj_data['obj_class'] == 'traffic_light' and frame_id in obj_data['frames']:
            index = obj_data['frames'].index(frame_id)
            context['traffic_light_type'] = obj_data['obj_type']
            context['traffic_light_state'] = obj_data['state'][index]
            break  # Assuming one traffic light per frame for simplicity, adjust as needed

    # Retrieve sign-related information
    for obj_id, obj_data in traffic_annotations.items():
        if obj_data['obj_class'] == 'sign' and frame_id in obj_data['frames']:
            context['sign_type'] = obj_data['obj_type']
            break  # Assuming one sign per frame for simplicity, adjust as needed

    # Check for crosswalk presence
    context['crosswalk_present'] = any(obj_data['obj_class'] == 'crosswalk' for obj_data in traffic_annotations.values())

    return context

def extract_set_vid_from_path(img_path):
    """
    Extracts set_id and vid_id from the image path.
    Assumes the path structure is './PIE_dataset/images/setXX/video_XXXX/XXXXX.png'.
    """
    path_parts = img_path.split('/')
    set_id = path_parts[-3]  # 'set01'
    vid_id = path_parts[-2]  # 'video_0001'
    return set_id, vid_id

def preprocess_data(data, pie_data_instance, annotations, image_size=(224, 224), num_frames=10, num_channels=3):
    images = []
    labels = []
    contexts = []  # Store context information
    set_vid_ids = []
    frame_ids_list = []  # Store frame IDs for verification

    for img_seq, label_seq, prob_seq in zip(data['image'], data['intention_binary'], data['intention_prob']):
        current_images = []
        current_contexts = []  # To store context per frame
        current_frame_ids = []  # To store frame IDs
        set_vid_pair = None

        for img_path, label, prob in zip(img_seq, label_seq, prob_seq):
            if os.path.isfile(img_path):
                img = cv2.imread(img_path)
                if img is None:  # Catch cases where the image could not be read
                    print(f"Warning: Could not read image {img_path}, skipping.")
                    continue

                # Extract set_id and vid_id
                if not set_vid_pair:
                    set_vid_pair = extract_set_vid_from_path(img_path)

                # Extract frame number from image path (e.g., '01018.png' -> 1018)
                frame_number = int(os.path.basename(img_path).split('.')[0])

                # Retrieve context for this frame
                context_info = get_context_for_frame(annotations, set_vid_pair[0], set_vid_pair[1], frame_number)
                current_contexts.append(context_info)
                current_frame_ids.append(frame_number)

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
        contexts.append(current_contexts)
        set_vid_ids.append(set_vid_pair)  # Store the set and vid ids
        frame_ids_list.append(current_frame_ids)  # Store frame IDs

        # Print to validate
        print(f"label[0]: {label[0]}, intention_prob: {prob[0]}, set_id: {set_vid_pair[0]}, vid_id: {set_vid_pair[1]}")
        print(f"Frame IDs: {current_frame_ids}")
        print(f"Context Info for first frame: {context_info}")

    images = np.array(images, dtype=np.float32) / 255.0  # Normalize
    labels = np.array(labels, dtype=np.float32)
    
    return images, labels, set_vid_ids, contexts, frame_ids_list

def split_data(images, labels, test_size=0.2, val_size=0.1):
    # Split into train+val and test
    X_train_val, X_test, y_train_val, y_test = train_test_split(images, labels, test_size=test_size, random_state=42)
    # Split train+val into train and val
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=val_size_adjusted, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test

import torch

def dynamic_threshold(context_tensor, base_threshold=0.5):
    """
    Adjusts the threshold dynamically based on the context tensor.
    
    Args:
    - context_tensor (torch.Tensor): Tensor containing contextual information.
    - base_threshold (float): The base decision threshold.

    Returns:
    - threshold (float): The adjusted threshold.
    """
    # Extract values from context_tensor based on their positions
    OBD_speed = context_tensor[0]
    ped_gesture = context_tensor[4]
    ped_look = context_tensor[5]
    ped_cross = context_tensor[6]
    crosswalk_present = context_tensor[10]
    
    threshold = base_threshold

    # High-risk situations (lower the threshold more aggressively)
    if OBD_speed > 20:  # High vehicle speed
        threshold -= 0.15
    if crosswalk_present == 1:  # Crosswalk is present
        threshold -= 0.1
    if ped_gesture > 0:  # Pedestrian showing a crossing gesture
        threshold -= 0.15
    if ped_look == 1:  # Pedestrian looking at the vehicle
        threshold -= 0.1
    
    # Low-risk situations (raise the threshold more conservatively)
    if OBD_speed < 5:  # Low vehicle speed
        threshold += 0.1
    if ped_look == 0:  # Pedestrian not looking at the vehicle
        threshold += 0.1
    if ped_cross in [-1, 0]:  # Pedestrian not crossing
        threshold += 0.15

    # Ensure threshold stays within a reasonable range
    threshold = max(0.1, min(threshold, 0.9))
    
    return threshold

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
