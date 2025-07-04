import json
from datasets import load_dataset, Features, Value, Sequence, Image
import base64
import os
from PIL import Image
import io


def get_base64_str():
    def _base64_str(pil):
        buffered = io.BytesIO()
        img = pil.convert("RGB")  
        img.save(buffered, format="JPEG")  
        base64_str = base64.standard_b64encode(buffered.getvalue()).decode("utf-8")
        return base64_str

    REPO_ID="hexuan21/VS2_raw_cmt"
    num=50

    data = load_dataset(REPO_ID, split="train")

    new_data=[]

    for i in range(num):
        sample=data[i]
        eg_frames=sample['eg_frames']
        new_data.append({
            "video_name": sample['video_name'],
            "prompt": sample['prompt'],
            "eg_frames_base64": [_base64_str(frame) for frame in eg_frames],
        })
        
    json.dump(new_data, open("examples_base64.json", "w", encoding="utf-8"), indent=4, ensure_ascii=True)

def build_few_shot():
    data2=json.load(open("few_shot_examples.json", "r", encoding="utf-8"))

    data=json.load(open("examples_base64.json", "r", encoding="utf-8"))

    for x in data:
        for i,y in enumerate(data2):
            video_name=x['video_name']
            if x['video_name']==y["video_name"]:
                data2[i]['frame_base64_list']=x['eg_frames_base64']
                print(f"FIND, for {video_name}")
                print(len(data2[i]['frame_base64_list']))
                print(len(data2[i]['frame_base64_list'][0]))
                break

    json.dump(data2, open("few_shot_examples_new.json", "w", encoding="utf-8"), indent=4, ensure_ascii=True)    
    
REPO_ID="hexuan21/VS2_raw_cmt_new"
# VIDEO_REPO_ID="hexuan21/vs2_sft_video"

dataset = load_dataset(REPO_ID, data_files="json_data/data_part_13.json", split="train")
print(dataset[0]['eg_frames']) 

dataset = load_dataset(REPO_ID, data_files="data/data_part_13.parquet", split="train")
print(dataset[0]['eg_frames']) 