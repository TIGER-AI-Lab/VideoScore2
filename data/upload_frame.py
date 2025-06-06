import os
import cv2
import numpy as np
import urllib.request
from huggingface_hub import upload_folder,login,HfApi
import json
from tqdm import tqdm


def fetch_upload_frames(video_url,video_name,save_dir,):
    
    video_path=os.path.join(save_dir,"videos",f"{video_name}.mp4")
    os.makedirs(os.path.dirname(video_path),exist_ok=True)
    if os.path.exists(video_path):
        return
    
    urllib.request.urlretrieve(video_url, video_path)    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_frames=None
    if total_frames<=24:
        n_frames=3
    elif total_frames<=64 and total_frames>24:
        n_frames=4
    else:
        n_frames=5
        
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
        frame_path = os.path.join(save_dir,"frames",video_name,f"{video_name}_{i}.jpg")
        if os.path.exists(frame_path):
            continue
        os.makedirs(os.path.dirname(frame_path),exist_ok=True)
        cv2.imwrite(frame_path, frame)
    
    
    
def upload(frames_dir):
    repo_id="hexuan21/VS2_frame_part_cache"
    
    api = HfApi()
    try:
        api.repo_info(repo_id=repo_id, repo_type="dataset",token=HF_TOKEN)
    except Exception as e:
        api.create_repo(
            repo_id=repo_id.split("/")[-1],
            repo_type="dataset",
            token=HF_TOKEN
        )
        print("📁 Created new repository:", repo_id)
        
    upload_folder(
        folder_path=frames_dir,         
        repo_id=repo_id,  
        repo_type="dataset",        
        path_in_repo="",                         
        commit_message="upload frames",
        token=HF_TOKEN
    )
    

if __name__ == "__main__":
    HF_TOKEN=os.environ["HF_TOKEN"]
    anno_path="batch5.json"
    n=500
    with open(anno_path,"r",encoding="utf-8") as f:
        raw_annos=json.load(f)[:n]
    
    current_dir=os.path.dirname(os.path.abspath(__file__))
    frame_temp_dir=os.path.join(current_dir,"video_frames")
    os.makedirs(frame_temp_dir,exist_ok=True)
    
    for anno in tqdm(raw_annos):
        url=anno["info"]["data"][2]["content"]
        video_name=url.split("/")[-1].split(".")[0]
        fetch_upload_frames(url,video_name,frame_temp_dir)
        
    upload(f"{frame_temp_dir}/frames")