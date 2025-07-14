import json
from datasets import load_dataset, Features, Value, Sequence, Image
import base64
import os
from PIL import Image
import io
from tqdm import tqdm

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
    


def split_batchs():

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
    data=json.load(open(path,"r",encoding='utf-8'))
    for batch_name, uid in dict1.items():
        new_data=[]
        for x in data:
            if str(x["batchId"]) == uid:
                new_data.append(x)
        print(f"{batch_name}, {len(new_data)}")
        with open(f"raw_anno/{batch_name}.json","w",encoding='utf-8') as f:
            json.dump(new_data,f,indent=4,ensure_ascii=False)


def check_hf_files():
    from huggingface_hub import list_repo_files
    repo_id = "hexuan21/vs2_raw_comment"
    files = list_repo_files(repo_id=repo_id, repo_type="dataset")
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

    fs=[f"{x.split('/')[1].split('.')[0]}.parquet" for x in anno_paths]
    for f in fs:
        target_file = f
        if target_file in files:
            print("✅ Found:", target_file)
        else:
            print("❌ Not found:", target_file)




def critical_modify():

    p="thinking_cmt/thinking_com_5k_original.json"
    new_p="thinking_cmt/thinking_com_5k_v1.json"
    data=json.load(open(p,"r"))

    excl_models=['stepvideo_t2v_low_vram']

    v_good_models=['lavie_base','magictime','cogvideox_5b']
    v_medium_models=['anidiff','cogvideox_2b','ltx_video_095',]
    v_bad_models=['ltx_video_091','latte','vchitect2',]

    p_good_models=['magictime',]
    p_medium_models=['ltx_video_095','cogvideox_5b','lavie_base']
    p_bad_models=['cogvideox_2b','anidiff','ltx_video_091','latte','vchitect2',]

    for idx, x in tqdm(enumerate(data)):
        url=x["video_url"]
        #"https://molar-public.oss-cn-hangzhou.aliyuncs.com/VideoScore/4500_4999/vchitect2/004501_p.mp4"
        t2v_model=url.split('/')[-2]
        v_score=x["visual_score"]
        t_score=x["t2v_score"]
        p_score=x["phy_score"]
        if t2v_model in v_bad_models:
            data[idx]["visual_score"]=min(2,v_score)
        if t2v_model in v_medium_models:
            data[idx]["visual_score"]=min(3,v_score)
        if t2v_model in v_good_models:
            data[idx]["visual_score"]=min(4,v_score)
        
        if t2v_model not in excl_models:
            data[idx]["t2v_score"]=min(4,t_score)
            
        if t2v_model in p_bad_models:
            data[idx]["phy_score"]=min(2,p_score)
        if t2v_model in p_medium_models:
            data[idx]["phy_score"]=min(3,p_score)
        if t2v_model in p_good_models:
            data[idx]["phy_score"]=min(4,p_score)
            
    with open(new_p,'w') as f:
        json.dump(data,f,indent=4,ensure_ascii=False)
     
     
p1="/data/xuan/videoscore2/temp/videos/"
p2="/data/xuan/videoscore2/videos_tmp_for_zip"
