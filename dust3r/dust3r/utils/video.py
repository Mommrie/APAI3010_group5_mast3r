import os
import cv2
import numpy as np
import PIL.Image
import tempfile
from tqdm import tqdm

def extract_video_frames(video_path, output_dir=None, max_frames=100, method='uniform', 
                         interval=15, min_scene_score=30.0, verbose=True):
    """
    从视频中提取帧并保存到指定目录
    
    参数:
        video_path: 视频文件路径
        output_dir: 输出目录，如果为None则创建临时目录
        max_frames: 最大提取帧数
        method: 提取方法 - 'uniform'(均匀采样), 'interval'(固定间隔), 'keyframe'(关键帧检测)
        interval: 当method='interval'时使用的帧间隔
        min_scene_score: 当method='keyframe'时使用的场景变化阈值
        verbose: 是否显示处理进度
        
    返回:
        frame_paths: 保存的帧文件路径列表
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"视频文件不存在: {video_path}")
        
    # 如果没有提供输出目录，创建临时目录
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix='video_frames_')
    else:
        os.makedirs(output_dir, exist_ok=True)
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"无法打开视频: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if verbose:
        print(f"视频信息: {os.path.basename(video_path)}, 帧数: {total_frames}, FPS: {fps:.2f}")
    
    frame_paths = []
    video_basename = os.path.basename(video_path).split('.')[0]
    
    # 根据不同方法选择提取帧的索引
    if method == 'uniform':
        # 均匀采样帧
        if total_frames <= max_frames:
            indices = range(total_frames)
        else:
            indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
            
    elif method == 'interval':
        # 按固定间隔提取帧
        step = interval
        indices = range(0, total_frames, step)
        # 限制最大帧数
        indices = list(indices)[:max_frames]
            
    elif method == 'keyframe':
        # 使用场景变化检测提取关键帧
        indices = []
        prev_frame = None
        
        if verbose:
            print("检测关键帧...")
            pbar = tqdm(total=total_frames)
            
        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
                
            if prev_frame is not None:
                # 计算帧间差异
                diff = cv2.absdiff(frame, prev_frame)
                score = np.sum(diff) / (diff.shape[0] * diff.shape[1] * diff.shape[2])
                
                if score > min_scene_score:
                    indices.append(i)
            
            prev_frame = frame
            
            if verbose:
                pbar.update(1)
                
        if verbose:
            pbar.close()
            
        # 如果检测到的关键帧太多，再次采样
        if len(indices) > max_frames:
            sample_indices = np.linspace(0, len(indices) - 1, max_frames, dtype=int)
            indices = [indices[i] for i in sample_indices]
        
        # 如果检测到的关键帧太少，补充均匀采样帧
        if len(indices) < max_frames:
            additional = max_frames - len(indices)
            existing_indices = set(indices)
            all_indices = set(range(total_frames))
            available_indices = list(all_indices - existing_indices)
            available_indices.sort()
            
            if len(available_indices) > additional:
                # 如果有足够多的可用帧，均匀采样
                extra_indices = np.linspace(0, len(available_indices) - 1, additional, dtype=int)
                extra_indices = [available_indices[i] for i in extra_indices]
            else:
                # 否则使用所有可用帧
                extra_indices = available_indices
                
            indices.extend(extra_indices)
            indices.sort()
    
    # 提取并保存所选帧
    saved_paths = []
    
    if verbose:
        print(f"提取 {len(indices)} 帧...")
        indices_iter = tqdm(indices)
    else:
        indices_iter = indices
        
    for i in indices_iter:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            # 转换BGR到RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 生成输出文件名
            output_filename = f"{video_basename}_frame_{i:06d}.jpg"
            output_path = os.path.join(output_dir, output_filename)
            # 保存图像
            PIL.Image.fromarray(frame_rgb).save(output_path)
            saved_paths.append(output_path)
    
    cap.release()
    
    if verbose:
        print(f"完成! 提取了 {len(saved_paths)} 帧，保存到: {output_dir}")
        
    return saved_paths

def process_video_for_reconstruction(video_path, max_frames=100, method='uniform', 
                                     interval=15, output_dir=None, verbose=True):
    """
    处理视频用于3D重建（便捷包装函数）
    
    返回:
        frame_paths: 帧文件路径列表，可直接传给现有重建流程
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