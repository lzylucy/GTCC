"""
    This file extracts video frames at 5Hz and generates DINOv2 or ResNet50 features for the egoprocel dataset.
    Modified to process a single handle passed as command-line argument.
    Added option to save frames and skip feature extraction.
    Added ResNet50 model for feature extraction with shape (N, 1024, 14, 14).
"""
import json
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import cv2
import tqdm
from torchvision import transforms, models
from PIL import Image
import gc
import argparse

# Get GPU ID from environment variable, default to 0
gpu_id = int(os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')[0])
device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')

print(f"Using device: {device}")

# Models will be loaded only if needed
dinov3_model = None
resnet50_model = None

def load_dinov3_model():
    """Lazy load DINOv3 model only when needed"""
    global dinov3_model
    if dinov3_model is None:
        dinov3_model = torch.hub.load("/local/real/liuzeyi/EgoVerse/dinov3", 
                               'dinov3_vits16', source='local', 
                               weights="/store/real/liuzeyi/ego-affordance/checkpoints/dinov3_vits16_pretrain_lvd1689m-08c60483.pth")
        dinov3_model.to(device)
        dinov3_model.eval()
    return dinov3_model

def load_resnet50_model():
    """Lazy load ResNet50 model only when needed"""
    global resnet50_model
    if resnet50_model is None:
        # Load pretrained ResNet50
        resnet = models.resnet50(pretrained=True)
        
        # Extract features after layer3 to get (N, 1024, 14, 14)
        # ResNet architecture: conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4, avgpool, fc
        resnet50_model = nn.Sequential(*list(resnet.children())[:-3])
        
        resnet50_model.to(device)
        resnet50_model.eval()
    return resnet50_model

# Image preprocessing for DINOv3
transform_dinov3 = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Image preprocessing for ResNet50
transform_resnet = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

image_process = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

def extract_frames_and_features(video_path, handle, batch_size=30, save_frames=False, 
                               skip_features=False, model_type='dinov3'):
    """
    Extract frames from video at 5Hz and optionally compute features.
    Returns features in (video_length, feature_dim) for dinov3 or (N, 1024, 14, 14) for resnet50.
    Process frames in chunks to avoid RAM exhaustion.
    
    Args:
        video_path: Path to video file
        handle: Handle name for saving frames
        batch_size: Number of frames to process at once
        save_frames: If True, save frames as numpy array
        skip_features: If True, skip feature extraction
        model_type: 'dinov3' or 'resnet50'
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return None
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, int(fps / 5))  # Extract every N frames for 5Hz
    
    # List to store frames if saving
    saved_frames = [] if save_frames else None
    
    # Don't store all frames in memory - process in chunks
    features_list = [] if not skip_features else None
    frame_count = 0
    extracted_frame_idx = 0
    frames_buffer = []
    
    frames_dir = '../videos/egoprocel/frames'
    os.makedirs(frames_dir, exist_ok=True)
    frames_path = os.path.join(frames_dir, f'{handle}.npy')
    if os.path.exists(frames_path):
        print(f"Frames already exist for {handle}, skipping frame extraction.")
        save_frames = False  # Don't overwrite existing frames
    
    print(f"Processing video: {total_frames} frames at {fps} FPS")
    print(f"Extracting every {frame_interval} frames (5Hz)")
    print(f"Model type: {model_type}")
    
    with tqdm.tqdm(total=total_frames, desc="Processing video", leave=False) as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                resized_image = image_process(Image.fromarray(frame_rgb))
                
                # Save frame to list if requested
                if save_frames:
                    saved_frames.append(np.array(resized_image))
                
                # Add to buffer for feature extraction if not skipping
                if not skip_features:
                    frames_buffer.append(frame_rgb)
                    
                    # Process buffer when it reaches batch_size to avoid RAM issues
                    if len(frames_buffer) >= batch_size:
                        batch_features = process_frame_batch(frames_buffer, model_type)
                        if batch_features is not None:
                            features_list.append(batch_features)
                        frames_buffer = []  # Clear buffer
                        gc.collect()  # Free memory
                
                extracted_frame_idx += 1
            
            frame_count += 1
            pbar.update(1)
    
    cap.release()
    
    print(f"Extracted {extracted_frame_idx} frames at 5Hz")
    
    # Save frames as numpy array if requested
    if save_frames and saved_frames:
        frames_array = np.array(saved_frames)
        np.save(frames_path, frames_array)
        print(f"Saved {len(saved_frames)} frames to {frames_path} with shape {frames_array.shape}")
        del saved_frames
        gc.collect()
    
    # If skipping features, return early
    if skip_features:
        return None
    
    # Process remaining frames in buffer
    if len(frames_buffer) > 0:
        batch_features = process_frame_batch(frames_buffer, model_type)
        if batch_features is not None:
            features_list.append(batch_features)
        frames_buffer = []
        gc.collect()
    
    if len(features_list) == 0:
        return None
    
    # Concatenate all features
    features_array = np.concatenate(features_list, axis=0)
    
    print(f"Features shape: {features_array.shape}")
    
    # Final cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    return features_array


def process_frame_batch(frames_list, model_type='dinov3'):
    """Process a batch of frames and return features"""
    try:
        # Load appropriate model
        if model_type == 'dinov3':
            model = load_dinov3_model()
            transform = transform_dinov3
        elif model_type == 'resnet50':
            model = load_resnet50_model()
            transform = transform_resnet
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        image_tensors = []
        for frame in frames_list:
            pil_image = Image.fromarray(frame)
            img_tensor = transform(pil_image)
            image_tensors.append(img_tensor)
        
        with torch.no_grad():
            images = torch.stack(image_tensors).to(device)
            batch_features = model(images)
            
            features = batch_features.detach().cpu().numpy()
            
            # Clean up
            del images, batch_features, image_tensors
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return features
    except Exception as e:
        print(f"Error processing batch: {e}")
        # Try to recover
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        return None


def process_handle(handle, task='pc_assembly', save_frames=False, skip_features=False, model_type='dinov3'):
    """Process a single handle"""
    path_to_features = f'../videos/egoprocel/features'
    os.makedirs(path_to_features, exist_ok=True)
    
    video_base_path = '/local/real/liuzeyi/GTCC/videos'
    
    # Check if already processed
    if not skip_features and os.path.exists(f'{path_to_features}/{handle}.npy'):
        print(f'Features already exist for {handle}, skipping.')
        return
    
    video_path = os.path.join(video_base_path, task, f'{handle}.mp4')
    
    if not os.path.exists(video_path):
        print(f'Video not found: {video_path}')
        return
    
    print(f'Processing video: {handle}')
    
    # Extract features
    features = extract_frames_and_features(video_path, handle, batch_size=30, 
                                          save_frames=save_frames, skip_features=skip_features,
                                          model_type=model_type)
    
    if not skip_features:
        if features is not None:
            # Save features
            np.save(f'{path_to_features}/{handle}.npy', features)
            print(f'Saved features for {handle}')
        else:
            print(f'Failed to extract features for {handle}')
    else:
        print(f'Skipped feature extraction for {handle}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract frames and features from video')
    parser.add_argument('handle', type=str, help='Video handle to process')
    parser.add_argument('--task', type=str, default='pc_assembly', help='Task name (default: pc_assembly)')
    parser.add_argument('--save-frames', action='store_true', help='Save frames to frames folder')
    parser.add_argument('--skip-features', action='store_true', help='Skip feature extraction')
    parser.add_argument('--model', type=str, default='dinov3', choices=['dinov3', 'resnet50'],
                       help='Model type for feature extraction (default: dinov3)')
    
    args = parser.parse_args()
    
    print(f'Processing handle: {args.handle}, task: {args.task}')
    print(f'Save frames: {args.save_frames}, Skip features: {args.skip_features}')
    print(f'Model: {args.model}')
    
    process_handle(args.handle, args.task, save_frames=args.save_frames, 
                  skip_features=args.skip_features, model_type=args.model)
    print('Done!')
