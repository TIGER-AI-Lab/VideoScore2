import json
import os
from string import Template
import urllib.request
import cv2
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from concurrent.futures import ThreadPoolExecutor, as_completed


def _fetch_video_single(video_name,video_url,f_v_save_dir):
    video_path = os.path.join(f_v_save_dir, "videos", f"{video_name}.mp4")
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    if not os.path.exists(video_path):
        try:
            urllib.request.urlretrieve(video_url, video_path)
            return video_path
        except Exception as e:
            print(f"[ERROR] Download failed for {video_name}: {e}")
            return None
    else:
        return video_path


def _fetch_frame_dir_single(video_name,video_url,f_v_save_dir):
    try:
        video_path=os.path.join(f_v_save_dir,"videos",f"{video_name}.mp4")
        os.makedirs(os.path.dirname(video_path),exist_ok=True)
        if not os.path.exists(video_path):
            urllib.request.urlretrieve(video_url, video_path)    
        
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        n_frames=None
        n_frames=int(total_frames // 6)
            
        frame_indices = np.linspace(0, total_frames - 1, num=n_frames, dtype=int)
        extracted_frames = []

        for idx in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            if idx in frame_indices:
                extracted_frames.append(frame)
        cap.release()
        
        for i, frame in enumerate(extracted_frames):
            frame_dir = os.path.join(f_v_save_dir,"frames",video_name)
            frame_abs_path = os.path.join(frame_dir,f"{video_name}_{i}.jpg")
            if os.path.exists(frame_abs_path):
                continue
            os.makedirs(os.path.dirname(frame_abs_path),exist_ok=True)
            cv2.imwrite(frame_abs_path, frame)  
        return frame_dir
    except Exception as e:
        print(f"❌ {video_name} failed: {e}")
        return None
    

def _fetch_frames_single(video_name,video_url,f_v_save_dir):
    try:
        video_path=os.path.join(f_v_save_dir,"videos",f"{video_name}.mp4")
        os.makedirs(os.path.dirname(video_path),exist_ok=True)
        if not os.path.exists(video_path):
            urllib.request.urlretrieve(video_url, video_path)    
        
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        n_frames=None
        n_frames=int(total_frames // 6)
            
        frame_indices = np.linspace(0, total_frames - 1, num=n_frames, dtype=int)
        extracted_frames = []

        for idx in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            if idx in frame_indices:
                extracted_frames.append(frame)
        cap.release()
        
        for i, frame in enumerate(extracted_frames):
            frame_dir = os.path.join(f_v_save_dir,"frames",video_name)
            frame_abs_path = os.path.join(frame_dir,f"{video_name}_{i}.jpg")
            if os.path.exists(frame_abs_path):
                continue
            os.makedirs(os.path.dirname(frame_abs_path),exist_ok=True)
            cv2.imwrite(frame_abs_path, frame)  
        return n_frames
    except Exception as e:
        print(f"❌ {video_name} failed: {e}")
        return None



def download_video_from_data(paths,f_v_save_dir, max_workers=8):
    data = []
    for path in paths:
        with open(path, "r", encoding='utf-8') as f:
            data.extend(json.load(f))
    print(f"Total videos to download: {len(data)}")

    video_names=[x['video_name'] for x in data]
    video_urls=[x['video_url'] for x in data]
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_fetch_video_single, name, url, f_v_save_dir) for name, url in zip(video_names,video_urls)]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading videos"):
            result = future.result()
            if result is None:
                continue  


def download_video_from_parquet(repo_id,parquet_names,f_v_save_dir,max_workers=8):
    data = load_dataset(repo_id, data_files=[f"{p_n}.parquet" for p_n in parquet_names],split="train")
    print(len(data))
    
    video_names=[x['video_name'] for x in data]
    video_urls=[x['video_url'] for x in data]
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_fetch_video_single, name, url, f_v_save_dir) for name, url in zip(video_names,video_urls)]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading videos"):
            result = future.result()
            if result is None:
                continue  


def download_frames_from_anno(anno_paths, f_v_save_dir, max_workers=8):
    all_annos = []
    for anno_path in anno_paths:
        with open(anno_path, "r", encoding="utf-8") as f:
            all_annos.extend(json.load(f))
    video_urls=[anno["info"]["data"][2]["content"] for anno in all_annos]
    video_names=[url.split("/")[-1].split(".")[0] for url in video_urls]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:     
  
        futures = [executor.submit(_fetch_frames_single, name, url, f_v_save_dir) for name, url in zip(video_names,video_urls)]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading frames"):
            result = future.result()
            

def download_frames_from_data(paths, f_v_save_dir, max_workers=8):
    data = []
    for path in paths:
        with open(path, "r", encoding='utf-8') as f:
            data.extend(json.load(f))
    print(f"Total videos to download: {len(data)}")
    
    video_names=[x['video_name'] for x in data]
    video_urls=[x['video_url'] for x in data]
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:     
  
        futures = [executor.submit(_fetch_frames_single, name, url, f_v_save_dir) for name, url in zip(video_names,video_urls)]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading frames"):
            result = future.result()
