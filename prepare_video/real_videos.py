

# yt-dlp -f "bestvideo[ext=mp4]+bestaudio[ext=mp4]" -o "my_video.mp4" --download-sections "*0:05:47.080-0:06:04.297" https://www.youtube.com/watch?v=D03BQb0sEqw


import json
import os
import subprocess
import ast
from tqdm import tqdm
import time
from datetime import datetime, timedelta

def download_koala_real_videos():
    root_dir="/data/xuan/videoscore2"
    temp_dir=os.path.join(root_dir,"videos/real_videos_koala_temp")
    os.makedirs(temp_dir,exist_ok=True)
    real_video_dir=os.path.join(root_dir,"videos/real_videos_koala")
    os.makedirs(real_video_dir,exist_ok=True)
    
    koala_raw_path=os.path.join(root_dir,"text_prompts","prompt_koala.jsonl")
    koala_raw_items=[json.loads(line.strip()) for line in open(koala_raw_path,"r")]
    srcID_stamp_mapping={}
    for item in koala_raw_items:
        # raw koala prompt item: 
        # {"src_id": "NZl1SLM0rtk_12", "timestamp": "['0:02:46.000', '0:02:51.599']", "text": "..."}
        srcID_stamp_mapping[item["src_id"]]=ast.literal_eval(item["timestamp"])
    
    prompt_path=os.path.join(root_dir,"text_prompts","all_prompts_en.jsonl")
    prompt_items=[json.loads(line.strip()) for line in open(prompt_path,"r")]

    
    for item in tqdm(prompt_items):
        # prompt_item: 
        # {"video_id": "000006", "text": "...", "src": "vidprom", "src_id": "59351a66-ce5b-5580-ac5f-93419f5e5861"}
        # {"video_id": "000013", "text": "...", "src": "koala", "src_id": "HVvPvxx1GfQ_8"}
        if item["src"]=="koala":
            video_id=item["video_id"]
            src_id=item["src_id"]
            time_stamp=srcID_stamp_mapping[src_id]
            start=datetime.strptime(time_stamp[0], "%H:%M:%S.%f")
            time_stamp[0]=(start+timedelta(seconds=0.5)).strftime("%H:%M:%S.%f")[:-3]
            # end=datetime.strptime(time_stamp[1], "%H:%M:%S.%f")
            # time_stamp[1]=(end-timedelta(seconds=0.5)).strftime("%H:%M:%S.%f")[:-3]
            
            # temp_path=os.path.join(temp_dir,f"{video_id}_R.mp4")
            
            output_path=os.path.join(real_video_dir,f"{video_id}_R.mp4")
            url=f"https://www.youtube.com/watch?v={src_id}"
            subprocess.run(f"yt-dlp {url} -f \"bestvideo[ext=mp4]\" -o {output_path} --download-sections \"*{time_stamp[0]}-{time_stamp[1]}\"",shell=True)            
            
            # subprocess.run(f"ffmpeg -i {temp_path} -an -c:v copy {output_path}",shell=True)
    
    
    
if __name__  == "__main__":
    # download_koala_real_videos()
    dir="/data/xuan/videoscore2/anno_page/anno_upload_shuffled"
    os.makedirs(dir,exist_ok=True)
    p="/data/xuan/videoscore2/anno_page/anno_upload/intra_inter_shuffled_all.json"
    with open(p,"r") as f:
        data=json.load(f)
    
    for i in range(100):
        chunk=data[i*500:(i+1)*500]
        with open(os.path.join(dir,f"batch_{i}.json"),"w") as f:
            json.dump(chunk,f,indent=4)
        print(f"chunk {i} done")