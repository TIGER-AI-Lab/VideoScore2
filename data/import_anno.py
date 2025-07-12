import os
from time import sleep
import cv2
import re
import numpy as np
import urllib.request
import json
from tqdm import tqdm
from huggingface_hub import upload_file,upload_folder
from datasets import Dataset, Features, Value, Sequence, Image





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



def download_frames(anno_paths,frame_temp_dir):
    for anno_path in anno_paths:
        with open(anno_path,"r",encoding="utf-8") as f:
            raw_annos=json.load(f)

        for anno in tqdm(raw_annos):
            url=anno["info"]["data"][2]["content"]
            video_name=url.split("/")[-1].split(".")[0]
            n_frames=_fetch_frames(url,video_name,frame_temp_dir)


def rebuild_rej_data(rej_data_path,batch_name,frame_temp_dir):
    rej_data=json.load(open(rej_data_path,"r"))
    data=[]
    for rej_item in rej_data:
        video_name=rej_item["video_name"]
        frame_names=os.listdir(os.path.join(frame_temp_dir,"frames",video_name))
        print(frame_names)
        frame_abs_paths=[os.path.join(frame_temp_dir,"frames",video_name,frame_name) for frame_name in frame_names]
        print(frame_abs_paths)
        exit()
        data_item={
            "video_name":video_name,
            "video_url":rej_item["video_url"],
            "prompt":rej_item["prompt"],
            "batch_name":batch_name,
            "visual_score":rej_item["visual_score"],
            "visual_comment_raw":rej_item["visual_cmt_raw"],
            "t2v_align_score":rej_item["t2v_score"],
            "t2v_align_comment_raw":rej_item["t2v_cmt_raw"],
            "phy_score":rej_item["phy_score"],
            "phy_comment_raw":rej_item["phy_cmt_raw"],
            "eg_frames":[{"bytes": open(p, "rb").read()} for p in frame_abs_paths]
        }
        data.append(data_item)
    
    ds = Dataset.from_list(data, features=FEATURES)
    local_parquet_dir= f"anno_parquet"
    os.makedirs(local_parquet_dir, exist_ok=True)
    parquet_name = f"{batch_name}.parquet"
    parquet_local_path = os.path.join(local_parquet_dir, parquet_name)
    ds.to_parquet(parquet_local_path)
    
    upload_file(
        path_or_fileobj=parquet_local_path,
        path_in_repo=parquet_name,
        repo_id=REPO_ID,
        repo_type="dataset",
        token=HF_TOKEN
    )



def build_raw_cmt_data(anno_local_paths,batch_name,frame_temp_dir):

    for anno_local_path,batch_name in zip(anno_local_paths,batch_names):
        upload_file(
            path_or_fileobj=anno_local_path,
            path_in_repo=f"raw_anno/{batch_name}.json",
            repo_id=REPO_ID,
            repo_type="dataset",
            token=HF_TOKEN
        )
        raw_annos=[]
        with open(anno_local_path,"r",encoding="utf-8") as f:
            raw_annos.extend(json.load(f))
            
        data=[]
        for anno in tqdm(raw_annos):
            url=anno["info"]["data"][2]["content"]
            video_name=url.split("/")[-1].split(".")[0]
            prompt_en=anno["info"]["data"][1]["content"].split("English Prompt")[1].split("\n")[0].strip(". :\n")
            try:
                visual_score=None
                t2v_score=None
                phy_score=None
                visual_cmt=None
                t2v_cmt=None
                phy_cmt=None
                for label_dict in anno["labels"]:
                    if "视觉质量评分" in label_dict["data"]["label"]:
                        visual_score=int(re.search(r'\d+', str(label_dict["data"]["value"])).group())
                    if "文本符合度评分" in label_dict["data"]["label"]:
                        t2v_score=int(re.search(r'\d+', str(label_dict["data"]["value"])).group())
                    if "物理符合度评分" in label_dict["data"]["label"]:
                        phy_score=int(re.search(r'\d+', str(label_dict["data"]["value"])).group())
                        
                    if "视觉质量描述" in label_dict["data"]["label"]:
                        visual_cmt=str(label_dict["data"]["value"])
                    if "文本符合度描述" in label_dict["data"]["label"]:
                        t2v_cmt=str(label_dict["data"]["value"])
                    if "物理符合度描述" in label_dict["data"]["label"]:
                        phy_cmt=str(label_dict["data"]["value"])
                    
                if visual_score is None:
                    raise ValueError(f"visual score not found for {video_name}")
                if t2v_score is None:
                    raise ValueError(f"t2v score not found for {video_name}")
                if phy_score is None:
                    raise ValueError(f"phy score not found for {video_name}")
                if visual_cmt is None:
                    raise ValueError(f"visual cmt not found for {video_name}")
                if t2v_cmt is None:
                    raise ValueError(f"t2v cmt not found for {video_name}")
                if phy_cmt is None:
                    raise ValueError(f"phy cmt not found for {video_name}")
                
            except Exception as e:
                print(e)
                continue
            
            if visual_score==MIN_SCORE:
                visual_cmt=SHARED_CMTS["visual_1"]
            if visual_score==MAX_SCORE:
                visual_cmt=SHARED_CMTS["visual_5"]
            if t2v_score==MAX_SCORE:
                t2v_cmt=SHARED_CMTS["t2v_5"]
            if phy_score==MAX_SCORE:
                phy_cmt=SHARED_CMTS["phy_5"]     
            
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
                    
            frame_abs_paths=[os.path.join(frame_temp_dir,"frames",video_name,f"{video_name}_{i}.jpg") for i in range(n_frames)]
            if not all(os.path.exists(p) for p in frame_abs_paths):
                print(f"not all frames exists for {video_name}, skipped\n")
                continue
            
            data_item={
                "video_name":video_name,
                "video_url":url,
                "prompt":prompt_en,
                "batch_name":batch_name,
                "visual_score":visual_score,
                "visual_comment_raw":visual_cmt,
                "t2v_align_score":t2v_score,
                "t2v_align_comment_raw":t2v_cmt,
                "phy_score":phy_score,
                "phy_comment_raw":phy_cmt,
                "eg_frames":[{"bytes": open(p, "rb").read()} for p in frame_abs_paths]
            }
            data.append(data_item)
        
        # json_file=f"data_part_{batch_id}.json"
        # with open(json_file,"w") as f:
        #     json.dump(data,f,indent=4)
        # upload_file(
        #     path_or_fileobj=json_file,
        #     path_in_repo=f"json_data/{json_file}",
        #     repo_id=REPO_ID,
        #     repo_type="dataset",
        #     token=HF_TOKEN
        # )
        
        ds = Dataset.from_list(data, features=FEATURES)
        local_parquet_dir= f"anno_parquet"
        os.makedirs(local_parquet_dir, exist_ok=True)
        parquet_name = f"{batch_name}.parquet"
        parquet_local_path = os.path.join(local_parquet_dir, parquet_name)
        ds.to_parquet(parquet_local_path)
        
        upload_file(
            path_or_fileobj=parquet_local_path,
            path_in_repo=parquet_name,
            repo_id=REPO_ID,
            repo_type="dataset",
            token=HF_TOKEN
        )
        


if __name__ == "__main__":
    MIN_SCORE=1
    MAX_SCORE=5
    REPO_ID="hexuan21/vs2_raw_comment"
    HF_TOKEN=os.environ["HF_TOKEN"]
    FEATURES = Features({
        "video_name":Value("string"),
        "video_url":Value("string"),
        "batch_name":Value("string"),
        "prompt":Value("string"),
        "visual_score":Value("int32"),
        "visual_comment_raw":Value("string"),
        "t2v_align_score":Value("int32"),
        "t2v_align_comment_raw":Value("string"),
        "phy_score":Value("int32"),
        "phy_comment_raw":Value("string"),
        "eg_frames": Sequence(feature=Image(decode=True)),
    })
    
    SHARED_CMTS={
        "visual_5": "High resolution, good clarity. No noticeable visual issues",
        "t2v_5": "Aligns well with prompt. Key elements are clearly represented.",
        "phy_5": "Good physical and commonsense consistency. No noticable issues.",
        "visual_1": "Low resolution and bad clarity. Local blurriness is present. Frequent visual distortions and misalignments. Abrupt and unsmooth transitions between adjacent frames. Unpolished and visually unstable, detracting from its watchability."
    }
    
    
    anno_paths=[
        f"raw_anno/com_5k.json",
        f"raw_anno/1.json",
        f"raw_anno/2.json",
        f"raw_anno/3.json",
        f"raw_anno/4.json",
        f"raw_anno/5.json",
        f"raw_anno/13.json",
        f"raw_anno/14.json",
        f"raw_anno/15.json",
        f"raw_anno/17.json",
        f"raw_anno/18.json",
        f"raw_anno/19.json",
        f"raw_anno/20.json",
        f"raw_anno/21.json",
        f"raw_anno/22.json",
        f"raw_anno/23.json",
        f"raw_anno/24.json",
        f"raw_anno/29.json",
        f"raw_anno/30.json",
        f"raw_anno/31.json",
        f"raw_anno/32.json",
        f"raw_anno/53.json",
        f"raw_anno/54.json",
        f"raw_anno/55.json",
        f"raw_anno/61.json",
        f"raw_anno/69.json",
        f"raw_anno/70.json"
    ]
    
    batch_names=[x.split('/')[1].split('.')[0] for x in anno_paths]

    frame_temp_dir="/data/xuan/videoscore2/temp"
    # download_frames(anno_paths,frame_temp_dir)
    build_raw_cmt_data(anno_paths,batch_names,frame_temp_dir)
    
    # rej_path="thinking_rejected/xxxx.json"
    # rej_batch_name="rej_xxxx"
    # rebuild_rej_data(rej_path,rej_batch_name,frame_temp_dir)
    
    
    
    
    
    
    
    
    [
        # "raw_anno/batch_91_100_com.json",
        
        # f"raw_anno/13.json",
        # f"raw_anno/14.json",
        # f"raw_anno/15.json",
        # f"raw_anno/17.json",
        # f"raw_anno/18.json",
        # f"raw_anno/19.json",
        # f"raw_anno/20.json",
        # f"raw_anno/21.json",
        # f"raw_anno/22.json",
        # f"raw_anno/23.json",
        # f"raw_anno/24.json",
        # f"raw_anno/29.json",
        # f"raw_anno/30.json",
        # f"raw_anno/31.json",
        # f"raw_anno/32.json"
        
        # f"raw_anno/37.json",
        # f"raw_anno/38.json",
        # f"raw_anno/39.json",
        # f"raw_anno/40.json",
        # f"raw_anno/45.json",
        # f"raw_anno/46.json",
        # f"raw_anno/47.json",
        # f"raw_anno/48.json",
        # f"raw_anno/53.json",
        # f"raw_anno/54.json",
        # f"raw_anno/55.json",
        # f"raw_anno/56.json",
        # f"raw_anno/61.json",
        # f"raw_anno/69.json",
        # f"raw_anno/70.json"
    ]