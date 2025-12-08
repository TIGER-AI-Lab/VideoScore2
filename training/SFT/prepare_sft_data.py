#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download SFT JSON + ZIP, update dataset_info.json, unzip videos.
Usage:
    python prepare_data.py --data_version_name data_27k_train_SFT
"""

import os
import shutil
import json
import argparse
import requests
from tqdm import tqdm
from zipfile import ZipFile, BadZipFile

HF_DATA_REPO   = "TIGER-Lab/VideoFeedback2"
CHUNK = 1 << 14  # 16 KB


def download_file(url: str, save_path: str, overwrite: bool = False, timeout: int = 15):
    if os.path.exists(save_path) and not overwrite:
        print(f"[skip] {save_path} already exists")
        return

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        bar = tqdm(total=total, unit="B", unit_scale=True, desc=os.path.basename(save_path))
        with open(save_path, "wb") as f:
            for chunk in r.iter_content(CHUNK):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
        bar.close()
    print(f"[ok] Downloaded → {save_path}")


def update_dataset_info(version_name: str, data_file: str, ds_info_path: str = "data/dataset_info.json"):
    os.makedirs(os.path.dirname(ds_info_path), exist_ok=True)

    if os.path.exists(ds_info_path):
        with open(ds_info_path, "r", encoding="utf-8") as f:
            ds_infos = json.load(f)
    else:
        ds_infos = {}
    
    ds_infos[version_name] = {
        "file_name": data_file,
        "formatting": "sharegpt",
        "columns": {
            "messages": "conversations",
            "videos": "videos"
        }
    }
        
    with open(ds_info_path, "w", encoding="utf-8") as f:
        json.dump(ds_infos, f, indent=2, ensure_ascii=False)
    print(f"[ok] dataset_info.json updated → {ds_info_path}")


def main(version_name: str):
    json_fname  = f"{version_name}.json"
    json_save_p = os.path.join("data", json_fname)
    base_name=version_name
    if "no_cot" in base_name:
        base_name=version_name.replace("_no_cot","")
    
    from huggingface_hub import list_repo_files
    hf_video_repo_files = list_repo_files(HF_DATA_REPO,repo_type="dataset")
        
    zip_file_list=[f for f in hf_video_repo_files if f.endswith(".zip") and "videos" in f]
    video_dir = os.path.join("data", "videos")
    print(zip_file_list)
    
    data_url = f"https://huggingface.co/datasets/{HF_DATA_REPO}/resolve/main/{json_fname}"
    download_file(data_url, json_save_p, overwrite=True)
    
    with open(json_save_p, "r", encoding="utf-8") as f:
        data = json.load(f)
    with open(json_save_p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    update_dataset_info(version_name, json_fname)
    
    
    for zip_file in zip_file_list:
        zip_url = f"https://huggingface.co/datasets/{HF_DATA_REPO}/resolve/main/{zip_file}"
        zip_save = os.path.join("data", f"{zip_file}")
        download_file(zip_url, zip_save, overwrite=True)
        os.makedirs(video_dir, exist_ok=True)
        try:
            with ZipFile(zip_save) as zf:
                zf.extractall(video_dir)
            print(f"[ok] Unzipped → {video_dir}")
        except BadZipFile as e:
            print(f"[error] Bad zip file: {e}")
        os.remove(zip_save)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare SFT JSON & ZIP")
    parser.add_argument("--data_version_name", required=True)
    args = parser.parse_args()
    main(args.data_version_name)

    # python prepare_sft_data.py --data_version_name data_27k_train_SFT
    