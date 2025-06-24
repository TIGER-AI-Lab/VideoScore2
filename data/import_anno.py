import os
from time import sleep
import cv2
import re
import numpy as np
import urllib.request
import json
from tqdm import tqdm
from huggingface_hub import upload_file
from datasets import Dataset, Features, Value, Sequence, Image



def _fetch_frames(video_url,video_name,save_dir,):
    
    video_path=os.path.join(save_dir,"videos",f"{video_name}.mp4")
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
        frame_path = os.path.join(save_dir,"frames",video_name,f"{video_name}_{i}.jpg")
        if os.path.exists(frame_path):
            continue
        os.makedirs(os.path.dirname(frame_path),exist_ok=True)
        cv2.imwrite(frame_path, frame)
    
    return n_frames



"""item format:
{
    "_id": "",
    "batchId": "",
    "info": {
        "data": [
            {
                "content": "PROMPT",
                "type": "TITLE"
            },
            {
                "content": "English Prompt: The sun rises over a serene city landscape, transitioning to bustling streets as fans in vibrant football jerseys converge towards iconic Premier League stadiums. The energy is palpable, with the excitement building for a day packed with football action. \n翻译为中文的Prompt:阳光照耀着宁静的城市风景，逐渐转向熙熙攘攘的街道，身穿鲜艳足球球衣的球迷们聚集向标志性的英超体育场。氛围令人振奋，兴奋感在为充满足球赛事的一天而不断升温。",
                "type": "TEXT"
            },
            {
                "content": "https://molar-public.oss-cn-hangzhou.aliyuncs.com/VideoScore/1000_1499/ltx_video_091/001023_i.mp4",
                "type": "VIDEO"
            },
            {
                "content": "src_id: f731fee5-c9f6-57da-a98f-03be1b16df73  src: vidprom",
                "type": "TEXT"
            }
        ]
    },
    "labels": [
        {
            "_id": "",
            "data": {
                "id": 1,
                "hash": "1_视觉质量评分",
                "label": "1_视觉质量评分",
                "value": 3, (or "value": "3-Medium",)
                "drawType": "QUESTION",
                "count": 1
            }
        },
        {
            "_id": "",
            "data": {
                "id": 2,
                "hash": "1_视觉质量描述",
                "label": "1_视觉质量描述",
                "value": "画质比较模糊，看不清城市风景，街道和标志性的体育场",
                "drawType": "QUESTION",
                "count": 1
            }
        },
        ......
    ]
},
"""



def download_frames(anno_path,num,frame_temp_dir):
    with open(anno_path,"r",encoding="utf-8") as f:
        if type(num) is int:
            raw_annos=json.load(f)[:num]
        else:
            raw_annos=json.load(f)

    for anno in tqdm(raw_annos):
        url=anno["info"]["data"][2]["content"]
        video_name=url.split("/")[-1].split(".")[0]
        n_frames=_fetch_frames(url,video_name,frame_temp_dir)



def build_raw_cmt_data(anno_path,batch_id,num,frame_temp_dir):
    with open(anno_path,"r",encoding="utf-8") as f:
        if type(num) is int:
            raw_annos=json.load(f)[:num]
        else:
            raw_annos=json.load(f)
    
    data=[]
    for anno in tqdm(raw_annos):
        url=anno["info"]["data"][2]["content"]
        video_name=url.split("/")[-1].split(".")[0]
        prompt_en=anno["info"]["data"][1]["content"].split("English Prompt")[1].split("\n")[0].strip(". :\n")
        try:
            # visual_score=anno["labels"][0]["data"]["value"][0]
            visual_score=int(re.search(r'\d+', str(anno["labels"][0]["data"]["value"])).group())
        except:
            print(f"visual score not found for {video_name}")
            continue
        try:
            # t2v_score=anno["labels"][2]["data"]["value"][0]
            t2v_score=int(re.search(r'\d+', str(anno["labels"][2]["data"]["value"])).group())
        except:
            print(f"t2v score not found for {video_name}")
            continue
        try:
            # phy_score=anno["labels"][4]["data"]["value"][0]
            phy_score=int(re.search(r'\d+', str(anno["labels"][4]["data"]["value"])).group())
        except:
            print(f"physical score not found for {video_name}")
            continue
        
        visual_cmt=anno["labels"][1]["data"]["value"]
        t2v_cmt=anno["labels"][3]["data"]["value"]
        phy_cmt=anno["labels"][5]["data"]["value"]
        if visual_score==MIN_SCORE:
            visual_cmt=shared_cmts["visual_1"]
        if visual_score==MAX_SCORE:
            visual_cmt=shared_cmts["visual_5"]
        if t2v_score==MAX_SCORE:
            t2v_cmt=shared_cmts["t2v_5"]
        if phy_score==MAX_SCORE:
            phy_cmt=shared_cmts["phy_5"]     
        
        num_try=0
        while True:
            if num_try>3:
                print(f"fetch frames for {video_name} failed")
                exit()
            try:
                n_frames=_fetch_frames(url,video_name,frame_temp_dir)
                break
            except Exception as e:
                print(f"fetch frames for {video_name} seems time out, sleeping for 60s")
                num_try+=1
                sleep(60)
                
        eg_frame_paths=[os.path.join(frame_temp_dir,"frames",video_name,f"{video_name}_{i}.jpg") for i in range(n_frames)]
        if not all(os.path.exists(p) for p in eg_frame_paths):
            print(f"not all frames exists for {video_name}, skipped\n")
            continue
        
        data_item={
            "video_name":video_name,
            "video_url":url,
            "prompt":prompt_en,
            "batch_id":batch_id,
            "visual_score":visual_score,
            "visual_comment_raw":visual_cmt,
            "t2v_align_score":t2v_score,
            "t2v_align_comment_raw":t2v_cmt,
            "phy_score":phy_score,
            "phy_comment_raw":phy_cmt,
            "eg_frames":eg_frame_paths
        }
        data.append(data_item)
    
    features = Features({
        "video_name":Value("string"),
        "video_url":Value("string"),
        "batch_id":Value("string"),
        "prompt":Value("string"),
        "visual_score":Value("int32"),
        "visual_comment_raw":Value("string"),
        "t2v_align_score":Value("int32"),
        "t2v_align_comment_raw":Value("string"),
        "phy_score":Value("int32"),
        "phy_comment_raw":Value("string"),
        "eg_frames": Sequence(Image()),
    })
    dataset = Dataset.from_list(data, features=features)
    
    dataset.push_to_hub(repo_id=REPO_ID,token=HF_TOKEN,private=False)
    
    with open("data_raw_cmt.json", "w") as f:
        json.dump(dataset.to_dict(), f, indent=2)
    upload_file(
        path_or_fileobj="data_raw_cmt.json",
        path_in_repo="data_raw_cmt.json",
        repo_id=REPO_ID,
        repo_type="dataset",
        token=HF_TOKEN
    )
    

if __name__ == "__main__":
    MIN_SCORE=1
    MAX_SCORE=5
    REPO_ID="hexuan21/VS2_raw_cmt"
    HF_TOKEN=os.environ["HF_TOKEN"]
    
    shared_cmts={
        "visual_5": "High resolution, good clarity. No noticeable local blurriness or distortion, transitions between frames are smooth without any abrupt visual artifacts. Good and consistent brightness and contrast. A pleasant and immersive viewing experience, no distracting visual issues",
        "t2v_5": "Aligns very well with prompt. All key elements (like characters, actions, environment and other details) are clearly represented.",
        "phy_5": "high physical and commonsense consistency. All actions, object interactions, and environmental dynamics unfold in a natural and believable manner. No signs of unrealistic and unnatural events, temporal glitches, or spatial anomalies.",
        "visual_1": "Low resolution and bad clarity. Local blurriness is present. Frequent visual distortions and misalignments. Abrupt and unsmooth transitions between adjacent frames. Unpolished and visually unstable, detracting from its watchability."
    }
    
    anno_path="batch_13.json"
    batch_id=13
    num="all"
    frame_temp_dir="/data/xuan/videoscore2/temp"
    download_frames(anno_path,num,frame_temp_dir)
    build_raw_cmt_data(anno_path,batch_id,num,frame_temp_dir)
    