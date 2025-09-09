import os
from time import sleep
import re
import json
from tqdm import tqdm
from huggingface_hub import upload_file,upload_folder
from datasets import Dataset, DatasetInfo, Features, Value, Sequence, Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils_fetch_f_v import _fetch_eg_frames_single, download_eg_frames_from_anno



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
    "eg_frames": Sequence(Value("binary")),
})

SHARED_CMTS={
    "visual_5": "High resolution, good clarity. No noticeable visual issues",
    "t2v_5": "Aligns well with prompt. Key elements are clearly represented.",
    "phy_5": "Good physical and commonsense consistency. No noticable issues.",
    "visual_1": "Low resolution and bad clarity. Local blurriness is present. Frequent visual distortions and misalignments. Abrupt and unsmooth transitions between adjacent frames. Unpolished and visually unstable, detracting from its watchability.",
    "t2v_1_alter":"The video content is completely inconsistent with the text prompt; there is a large discrepancy.",
    "phy_1_alter":"The video content has very serious inconsistencies with common sense and physical laws.",
}


def _build_raw_cmt_single(anno,batch_name,f_v_save_dir):
    try:
        url = anno["info"]["data"][2]["content"]
        video_name = url.split("/")[-1].split(".")[0]
        prompt_en = (
            anno["info"]["data"][1]["content"]
            .split("English Prompt", 1)[1]
            .split("\n", 1)[0]
            .strip(". :\n")
        )
        # prompt_src=anno["info"]["data"][3]["content"].split("src:")[1].strip()
        # prompt_srcid=anno["info"]["data"][3]["content"].split("src:")[0].split("src_id:")[1].strip()
        
        visual_score = t2v_score = phy_score = None
        visual_cmt = t2v_cmt = phy_cmt = None
        if anno["labels"] == []:
            raise ValueError("no annotation")
        for label_dict in anno["labels"]:
            label = label_dict["data"]["label"]
            value = str(label_dict["data"]["value"])
            if "视觉质量评分" in label:
                visual_score = int(re.search(r"\d+", value).group())
            elif "文本符合度评分" in label:
                t2v_score = int(re.search(r"\d+", value).group())
            elif "物理符合度评分" in label:
                phy_score = int(re.search(r"\d+", value).group())
            elif "视觉质量描述" in label:
                visual_cmt = value
            elif "文本符合度描述" in label:
                t2v_cmt = value
            elif "物理符合度描述" in label:
                phy_cmt = value

        if None in (visual_score, t2v_score, phy_score, visual_cmt, t2v_cmt, phy_cmt):
            raise ValueError("missing score or comment")

        
        if visual_score == MAX_SCORE:
            visual_cmt = SHARED_CMTS["visual_5"]
        if t2v_score == MAX_SCORE:
            t2v_cmt = SHARED_CMTS["t2v_5"]
        if phy_score == MAX_SCORE:
            phy_cmt = SHARED_CMTS["phy_5"]
        if visual_score == MIN_SCORE:
            visual_cmt = SHARED_CMTS["visual_1"]
            
        # some annotators skip comments when t2v=1 or phy=1, the code below is to avoid empty comment
        if t2v_score==MIN_SCORE and len(str(t2v_cmt).strip())<=2:
            t2v_cmt=SHARED_CMTS["t2v_1_alter"]
        if phy_score==MIN_SCORE and len(str(phy_cmt).strip())<=2:
            phy_cmt=SHARED_CMTS["phy_1_alter"]
        
        num_try = 0
        while True:
            try:
                frame_dir_abs_path = _fetch_eg_frames_single(video_name, url, f_v_save_dir)
                n_frames = len(os.listdir(frame_dir_abs_path))
                break
            except Exception:
                num_try += 1
                if num_try > 3:
                    raise RuntimeError("frame fetch failed 3 times")
                sleep(60)

        eg_frame_abs_paths = [
            os.path.join(f_v_save_dir, "frames_eg", video_name, f"{video_name}_{i}.jpg")
            for i in range(n_frames)
        ]
        if not all(os.path.exists(p) for p in eg_frame_abs_paths):
            raise FileNotFoundError("some frames missing")

        return {
            "video_name": video_name,
            "video_url": url,
            "batch_name": batch_name,
            "prompt": prompt_en,
            "visual_score": visual_score,
            "visual_comment_raw": visual_cmt,
            "t2v_align_score": t2v_score,
            "t2v_align_comment_raw": t2v_cmt,
            "phy_score": phy_score,
            "phy_comment_raw": phy_cmt,
            "eg_frames": [open(p, "rb").read() for p in eg_frame_abs_paths],
        }
    except Exception as e:
        print(f"[ERROR] build item failed for {video_name}: {e}")
        return None


def build_raw_cmt_data(
    anno_local_paths, batch_names, f_v_save_dir, max_workers=8
):
    for anno_local_path, batch_name in zip(anno_local_paths, batch_names):
        upload_file(
            path_or_fileobj=anno_local_path,
            path_in_repo=f"anno_raw/{batch_name}.json",
            repo_id=REPO_ID,
            repo_type="dataset",
            token=HF_TOKEN,
        )

        with open(anno_local_path, "r", encoding="utf-8") as f:
            raw_annos = json.load(f)

        data = []
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [
                ex.submit(_build_raw_cmt_single, anno, batch_name, f_v_save_dir)
                for anno in raw_annos
            ]
            for fut in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"Processing {batch_name}",
            ):
                item = fut.result()
                if item is not None:
                    data.append(item)
        
        ds = Dataset.from_list(data, features=FEATURES,info=DatasetInfo(features=FEATURES))
        out_dir= f"anno_parquet"
        os.makedirs(out_dir, exist_ok=True)
        f_name = f"{batch_name}.parquet"
        f_path = os.path.join(out_dir, f_name)
        ds.to_parquet(f_path)
        
        res=upload_file(
            path_or_fileobj=f_path,
            path_in_repo=f_name,
            repo_id=REPO_ID,
            repo_type="dataset",
            token=HF_TOKEN
        )
        print(f"[OK] Uploaded {f_name} → {res}")
        

def rebuild_rej_data(rej_data_path,batch_name,f_v_save_dir):
    with open(rej_data_path,"r") as f:
        rej_data=json.load(f)
    data=[]
    for rej_item in rej_data:
        video_name=rej_item["video_name"]
        video_url=rej_item["video_url"]
        frame_names=os.listdir(os.path.join(f_v_save_dir,"frames_eg",video_name))
        eg_frame_abs_paths=[os.path.join(f_v_save_dir,"frames_eg",video_name,frame_name) for frame_name in frame_names]
        if not all(os.path.exists(p) for p in eg_frame_abs_paths):
            print(f"Some frames are missing for video {video_name}")
            _fetch_eg_frames_single(video_name,video_url,f_v_save_dir)
            
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
            "eg_frames":[{"bytes": open(p, "rb").read()} for p in eg_frame_abs_paths]
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



if __name__ == "__main__":
    
    f_v_save_dir="/data/xuan/data/videoscore2/f_v_all"
    
    anno_paths=[
        f"anno_raw/9.json",
        f"anno_raw/16.json",
        f"anno_raw/33.json",
        f"anno_raw/34.json",
        f"anno_raw/38.json",
        f"anno_raw/45.json",
        f"anno_raw/46.json",
        f"anno_raw/47.json",
        f"anno_raw/48.json",
        f"anno_raw/56.json",
        f"anno_raw/62.json",
        f"anno_raw/71.json",
        f"anno_raw/74.json",
        f"anno_raw/75.json",
        f"anno_raw/78.json",
        f"anno_raw/79.json",
        f"anno_raw/81.json",
        f"anno_raw/82.json",
        f"anno_raw/83.json",
        f"anno_raw/85.json",
        f"anno_raw/86.json",
    ]
    
    # download_eg_frames_from_anno(anno_paths,f_v_save_dir,max_workers=8)
    
    
    batch_names=[x.split('/')[1].split('.')[0] for x in anno_paths]
    build_raw_cmt_data(anno_paths,batch_names,f_v_save_dir)
    

    
    
    # rej_path="thinking_rejected/xxxx.json"
    # rej_batch_name="rej_xxxx"
    # rebuild_rej_data(rej_path,rej_batch_name,frame_temp_dir)
    
    
    # anno_paths=[
    #     f"anno_raw/com_5k_0.json",
    #     f"anno_raw/com_5k_1.json",
    #     f"anno_raw/com_5k_2.json",
    #     f"anno_raw/com_5k_3.json",
    #     f"anno_raw/com_5k_4.json",
    #     f"anno_raw/1.json",
    #     f"anno_raw/2.json",
    #     f"anno_raw/3.json",
    #     f"anno_raw/4.json",
    #     f"anno_raw/5.json",
    #     f"anno_raw/13.json",
    #     f"anno_raw/14.json",
    #     f"anno_raw/15.json",
    #     f"anno_raw/17.json",
    #     f"anno_raw/18.json",
    #     f"anno_raw/19.json",
    #     f"anno_raw/20.json",
    #     f"anno_raw/21.json",
    #     f"anno_raw/22.json",
    #     f"anno_raw/23.json",
    #     f"anno_raw/24.json",
    #     f"anno_raw/29.json",
    #     f"anno_raw/30.json",
    #     f"anno_raw/31.json",
    #     f"anno_raw/32.json",
    #     f"anno_raw/53.json",
    #     f"anno_raw/54.json",
    #     f"anno_raw/55.json",
    #     f"anno_raw/61.json",
    #     f"anno_raw/69.json",
    #     f"anno_raw/70.json"
    # ]
    
    
    
        
    
