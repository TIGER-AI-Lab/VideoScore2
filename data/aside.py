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
    
# REPO_ID="hexuan21/VS2_raw_cmt_new"
# # VIDEO_REPO_ID="hexuan21/vs2_sft_video"

# dataset = load_dataset(REPO_ID, data_files="json_data/data_part_13.json", split="train")
# print(dataset[0]['eg_frames']) 

# dataset = load_dataset(REPO_ID, data_files="data/data_part_13.parquet", split="train")
# print(dataset[0]['eg_frames']) 

# repo_id="hexuan21/VS2_raw_cmt"
# batch_name="batch_91_100_com"
# data = load_dataset(repo_id, data_files=f"{batch_name}.parquet",split="train")


# from huggingface_hub import upload_file
# parquet_local_path="parquet/batch_91_100_com.parquet"
# parquet_name = f"{batch_name}.parquet"
# repo_id="hexuan21/VS2_raw_cmt"
# batch_name="batch_91_100_com"
# HF_TOKEN="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"
# upload_file(
#         path_or_fileobj=parquet_local_path,
#         path_in_repo=parquet_name,
#         repo_id=repo_id,
#         repo_type="dataset",
#         token=HF_TOKEN
#     )


dict1={
13:
"67ff8a3c97cbd9edfc8fc4c2",
14:
"67ff8a3c97cbd9edfc8fc6b7",
15:
"67ff8a3c97cbd9edfc8fc8ac",
17:
"67ff8a3c97cbd9edfc8fcc96",
18:
"67ff8a3c97cbd9edfc8fce8b",
19:
"67ff8a3c97cbd9edfc8fd080",
20:
"67ff8a3c97cbd9edfc8fd275",
21:
"67ff8a3c97cbd9edfc8fd46a",
22:
"67ff8a3c97cbd9edfc8fd65f",
23:
"67ff8a3c97cbd9edfc8fd854",
24:
"67ff8a3c97cbd9edfc8fda49",
29:
"67ff8a3c97cbd9edfc8fe412",
30:
"67ff8a3c97cbd9edfc8fe607",
31:
"67ff8a3c97cbd9edfc8fe7fc",
32:
"67ff8a3c97cbd9edfc8fe9f1",
37:
"67ff8a3c97cbd9edfc8ff3ba",
38:
"67ff8a3c97cbd9edfc8ff5af",
39:
"67ff8a3c97cbd9edfc8ff7a4",
40:
"67ff8a3c97cbd9edfc8ff999",
45:
"67ff8a3c97cbd9edfc900362",
46:
"67ff8a3c97cbd9edfc900557",
47:
"67ff8a3c97cbd9edfc90074c",
48:
"67ff8a3c97cbd9edfc900941",
53:
"67ff8a3c97cbd9edfc90130a",
54:
"67ff8a3c97cbd9edfc9014ff",
55:
"67ff8a3c97cbd9edfc9016f4",
56:
"67ff8a3c97cbd9edfc9018e9",
61:
"67ff8a3c97cbd9edfc9022b2",
69:
"67ff8a3c97cbd9edfc90325a",
70:
"67ff8a3c97cbd9edfc90344f",

}

path="VideoScore2.json"

# data=json.load(open(path,"r",encoding='utf-8'))
# for batch_name, uid in dict1.items():
#     new_data=[]
#     for x in data:
#         if str(x["batchId"]) == uid:
#             new_data.append(x)
            
#     print(f"{batch_name}, {len(new_data)}")

#     with open(f"raw_anno/{batch_name}.json","w",encoding='utf-8') as f:
#         json.dump(new_data,f,indent=4,ensure_ascii=False)


# p="/home/brantley/workdir/VideoScore2/data/thinking_cmt/thinking_batch_91_100_com.json"
# data=json.load(open(p,"r",encoding="utf-8"))

# data=data[:20]

# with open("new.json","w") as f:
#     json.dump(data,f,indent=4,ensure_ascii=False)


