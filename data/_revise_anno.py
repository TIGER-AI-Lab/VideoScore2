import json
from tqdm import tqdm
import os
import re
from _analyze import plot
import random

EXCL_MODELS=['stepvideo_t2v','stepvideo_t2v_low_vram','kling','sora','wanx21_14b','ruyi',]

V_GOOD_MODELS=['lavie_base','magictime','cogvideox_5b','wanx21_1_3b','videocrafter2','opensora_plan_v1_3','pika_v2_2','cogvideox15_5b',]
V_MEDIUM_MODELS=['anidiff','cogvideox_2b','ltx_video_095','hotshot_xl','mochi1_preview','opensora_v1_2','latte',]
V_BAD_MODELS=['ltx_video_091','vchitect2','text2video_zero','modelscope','zeroscope']

P_GOOD_MODELS=['magictime','wanx21_1_3b','mochi1_preview','opensora_plan_v1_3','pika_v2_2','cogvideox15_5b',]
P_MEDIUM_MODELS=['ltx_video_095','cogvideox_5b','lavie_base','hotshot_xl','videocrafter2','latte',]
P_BAD_MODELS=['opensora_v1_2','cogvideox_2b','anidiff','ltx_video_091','vchitect2','text2video_zero','modelscope','zeroscope']


def modify_score(t2v_model,v_score,t_score,p_score):
    if t2v_model in V_BAD_MODELS:
        v_score_new=min(2,v_score)
    if t2v_model in V_MEDIUM_MODELS:
        v_score_new=min(3,v_score)
    if t2v_model in V_GOOD_MODELS:
        v_score_new=min(4,v_score)
    
    if t2v_model not in EXCL_MODELS:
        t_score_new=min(4,t_score)
        
    if t2v_model in P_BAD_MODELS:
        p_score_new=min(2,p_score)
    if t2v_model in P_MEDIUM_MODELS:
        p_score_new=min(3,p_score)
    if t2v_model in P_GOOD_MODELS:
        p_score_new=min(4,p_score)
    
    return v_score_new,t_score_new,p_score_new


def critical_modify_tk(paths,new_p,batch_name):
    data=[]
    for path in paths:
        with open(path,"r") as f:
            data.extend(json.load(f))
    
    # random.shuffle(data)
    
    for idx, x in tqdm(enumerate(data)):
        url=x["video_url"]
        #"https://molar-public.oss-cn-hangzhou.aliyuncs.com/VideoScore/4500_4999/vchitect2/004501_p.mp4"
        t2v_model=url.split('/')[-2]
        v_score=x["visual_score"]
        t_score=x["t2v_score"]
        p_score=x["phy_score"]
        
        data[idx]["visual_score"], data[idx]["t2v_score"], data[idx]["phy_score"]=modify_score(t2v_model,v_score,t_score,p_score)
    
    v_scores=[xx['visual_score'] for xx in data]
    t_scores=[xx['t2v_score'] for xx in data]
    p_scores=[xx['phy_score'] for xx in data]
    plot(v_scores,batch_name,1)
    plot(t_scores,batch_name,2)
    plot(p_scores,batch_name,3)
    
    with open(new_p,'w') as f:
        json.dump(data,f,indent=4,ensure_ascii=False)
     
    
    
def critical_modify_raw(paths,new_path,batch_name):
    data=[]
    for path in paths:
        with open(path,"r") as f:
            data.extend(json.load(f))
    new_data=[]
    for idx, raw_item in tqdm(enumerate(data)):
        x={}
        prompt= (
            raw_item["info"]["data"][1]["content"]
            .split("English Prompt", 1)[1]
            .split("\n", 1)[0]
            .strip(". :\n")
        )
        url = raw_item["info"]["data"][2]["content"]
        video_name = url.split("/")[-1].split(".")[0]
        #"https://molar-public.oss-cn-hangzhou.aliyuncs.com/VideoScore/4500_4999/vchitect2/004501_p.mp4"
        t2v_model=url.split('/')[-2]
        for label_dict in raw_item["labels"]:
            label = label_dict["data"]["label"]
            value = str(label_dict["data"]["value"])
            if "视觉质量评分" in label:
                v_score = int(re.search(r"\d+", value).group())
            elif "文本符合度评分" in label:
                t_score = int(re.search(r"\d+", value).group())
            elif "物理符合度评分" in label:
                p_score = int(re.search(r"\d+", value).group())
            elif "视觉质量描述" in label:
                visual_cmt = value
            elif "文本符合度描述" in label:
                t2v_cmt = value
            elif "物理符合度描述" in label:
                phy_cmt = value
        
        if p_score in [3,4,5] and t2v_model not in EXCL_MODELS:
            p_score-=1
        x['video_name']=video_name 
        x['video_url']=url
        x['prompt']=prompt
        x['visual_score']=v_score
        x['t2v_score']=t_score
        x['phy_score']=p_score
        x['visual_cmt_raw']=visual_cmt
        x['t2v_cmt_raw']=t2v_cmt
        x['phy_cmt_raw']=phy_cmt
        x["visual_score_old"]=v_score
        x["t2v_score_old"]=t_score
        x["phy_score_old"]=p_score
        
        x["visual_score"],x["visual_score"],x["visual_score"]=modify_score(t2v_model,v_score,t_score,p_score)
        
        new_data.append(x)
        
    v_scores=[xx['visual_score'] for xx in new_data]
    t_scores=[xx['t2v_score'] for xx in new_data]
    p_scores=[xx['phy_score'] for xx in new_data]
    plot(v_scores,batch_name,1)
    plot(t_scores,batch_name,2)
    plot(p_scores,batch_name,3)
    with open(new_path,'w') as f:
        json.dump(new_data,f,indent=4,ensure_ascii=False)
    
    
    
if __name__ == "__main__":
    # batch_idx="48"
    # paths=[
    #     f"anno_raw/{batch_idx}.json"
    # ]
    # new_temp_path=f"temp/{batch_idx}_raw_modified.json"   
    # batch_name=f"{batch_idx}_raw_modified"     
    # critical_modify_raw(paths,new_temp_path,batch_name)

    paths=[f"thinking_cmt_original/{fname}" for fname in os.listdir("thinking_cmt_original")]
    new_path="thinking_cmt/sft_17k_modified.json"
    batch_name="sft_17k_stage1"
    critical_modify_tk(paths,new_path,batch_name)