import json
import os

real_video_dir="/data/xuan/videoscore2/videos/real_videos_koala"
prompt_path="/data/xuan/videoscore2/text_prompts/all_prompts_en.jsonl"
shared_comments=json.load(open("./const/shared_comments.json","r"))
MAX_SCORE=5

def assign_scores_real(save_path):
    with open(prompt_path,"r") as f:
        prompt_items=[json.loads(line.strip()) for line in f]
        
    data=[]
    for item in prompt_items:
        if item["src"]!="koala":
            continue
        video_id=item["video_id"]
        prompt_en=item["text"]
        video_path=os.path.join(real_video_dir,f"{video_id}_R.mp4")
        if not os.path.exists(video_path):
            continue
        data_item={
            "video_name":f"{video_id}_R",
            "prompt":prompt_en,
            "visual":{
                "score":MAX_SCORE,
                "comment":shared_comments["visual_5"],
            },
            "t2v_align":{
                "score":MAX_SCORE,
                "comment":shared_comments["t2v_5"],
            },
            "physical":{
                "score":MAX_SCORE,
                "comment":shared_comments["physical_5"],
            }
        }
        data.append(data_item)
        
    # with open(save_path,"a") as f:
    #     json.dump(data,f,indent=4)
