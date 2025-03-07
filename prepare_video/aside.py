import json
import os
import shutil
from tqdm import tqdm


def copy_video_anno_try():
    t2v_models=["anidiff","gen2","latte","lavie_base","lvdm","magictime","morph_studio","open_sora_v_1_2","open_sora_v_1_1","vc2"]
    model_codes=['a','b','c','d','e','f','g','h','i','j']

    root_dir="/data/xuan/video_eval/leader_bd/videos"
    dest_dir="/data/xuan/videoscore2/anno_page/videoscore2_anno/videos_display"

    start=50
    num_video=50

    for model,model_code in zip(t2v_models,model_codes):
        video_dir=f"{root_dir}/{model}"
        for video in tqdm(sorted(os.listdir(video_dir))[start:start+num_video]):
            dest_idx=int(video.split("_")[1])
            dest_path=f"{dest_dir}/{dest_idx:06d}_{model_code}.mp4"
            shutil.copy(f"{video_dir}/{video}",dest_path)
        
    print(sorted(os.listdir(dest_dir))[:10])

    L=[]
    for i in range(start,start+num_video):
        for model_code in model_codes:
            L.append(f"{i:06d}_{model_code}") 
    
    with open("/data/xuan/videoscore2/anno_page/text_files/sampled_id.json","w") as f:
        json.dump({"video_ids":L},f,indent=4)

p="/data/xuan/videoscore2/text_prompts/prompt_en_koala.jsonl"
with open(p,"r") as f:
    data=[json.loads(line) for line in f]

for i, x in enumerate(data):
    if "." not in x["text"][-3:]:
        data[i]["text"]+="."
        
with open(p,"w") as f2:
    for p in data:
        f2.write(json.dumps(p)+"\n")

