import os
import cv2
import numpy as np
import PIL.Image
import tempfile
from tqdm import tqdm

def extract_video_frames(video_path, output_dir=None, max_frames=100, method='uniform', 
                         interval=15, min_scene_score=30.0, verbose=True):
    """
    Extract frames from the video and save them to the specified directory

    Parameters:
    video_path: Path to the video file
    output_dir: Output directory, if None a temporary directory will be created
    max_frames: Maximum number of frames to extract
    method: Extraction method - 'uniform' (uniform sampling), 'interval' (fixed interval), 'keyframe' (keyframe detection)
    interval: Frame interval used when method='interval'
    min_scene_score: Scene change threshold used when method='keyframe'
    verbose: Whether to display processing progress

    Return:
    saved_paths: List of saved frame file paths
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file does not exist: {video_path}")
        
    # If no output directory is provided, create a temporary directory
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix='video_frames_')
    else:
        os.makedirs(output_dir, exist_ok=True)
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Unable to open video: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if verbose:
        print(f"Video information: {os.path.basename(video_path)}, Frames: {total_frames}, FPS: {fps:.2f}")
    
    frame_paths = []
    video_basename = os.path.basename(video_path).split('.')[0]
    
    # Select the index of extracted frames according to different methods
    if method == 'uniform':
        # Uniform sampling frame
        if total_frames <= max_frames:
            indices = range(total_frames)
        else:
            indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
            
    elif method == 'interval':
        # Extract frames at fixed intervals
        step = interval
        indices = range(0, total_frames, step)
        # Limit maximum frame rate
        indices = list(indices)[:max_frames]
            
    elif method == 'keyframe':
        # Using scene change detection to extract key frames yields very poor results; it is recommended not to use it
        indices = []
        prev_frame = None
        
        if verbose:
            print("Detecting keyframes...")
            pbar = tqdm(total=total_frames)
            
        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
                
            if prev_frame is not None:
                # Calculate inter-frame difference
                diff = cv2.absdiff(frame, prev_frame)
                score = np.sum(diff) / (diff.shape[0] * diff.shape[1] * diff.shape[2])
                
                if score > min_scene_score:
                    indices.append(i)
            
            prev_frame = frame
            
            if verbose:
                pbar.update(1)
                
        if verbose:
            pbar.close()
            
        # If too many keyframes are detected, resample again
        if len(indices) > max_frames:
            sample_indices = np.linspace(0, len(indices) - 1, max_frames, dtype=int)
            indices = [indices[i] for i in sample_indices]
        
        # If too few keyframes are detected, supplement with evenly sampled frames
        if len(indices) < max_frames:
            additional = max_frames - len(indices)
            existing_indices = set(indices)
            all_indices = set(range(total_frames))
            available_indices = list(all_indices - existing_indices)
            available_indices.sort()
            
            if len(available_indices) > additional:
                # If there are enough available frames, sample evenly
                extra_indices = np.linspace(0, len(available_indices) - 1, additional, dtype=int)
                extra_indices = [available_indices[i] for i in extra_indices]
            else:
                # Otherwise, use all available frames
                extra_indices = available_indices
                
            indices.extend(extra_indices)
            indices.sort()
    
    # Extract and save the selected frame
    saved_paths = []
    
    if verbose:
        print(f"Extracting {len(indices)} frames...")
        indices_iter = tqdm(indices)
    else:
        indices_iter = indices
        
    for i in indices_iter:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            output_filename = f"{video_basename}_frame_{i:06d}.jpg"
            output_path = os.path.join(output_dir, output_filename)
            
            PIL.Image.fromarray(frame_rgb).save(output_path)
            saved_paths.append(output_path)
    
    cap.release()
    
    if verbose:
        print(f"Completed! Extracted {len(saved_paths)} frames, saved to: {output_dir}")
        
    return saved_paths

def process_video_for_reconstruction(video_path, max_frames=100, method='uniform', 
                                     interval=15, output_dir=None, verbose=True):
    """
    Process video for 3D reconstruction (wrapper function)

    Return:
        frame_paths: list of frame file paths, can be directly passed to the existing reconstruction pipeline
    """
    frame_paths = extract_video_frames(
        video_path, 
        output_dir=output_dir,
        max_frames=max_frames, 
        method=method,
        interval=interval,
        verbose=verbose
    )
    return frame_paths