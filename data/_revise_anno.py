import json
from tqdm import tqdm
import os
import re
from _analyze import plot
import random

EXCL_MODELS=['stepvideo_t2v','stepvideo_t2v_low_vram','kling','sora','wanx21_14b','ruyi',]

V_MAX4_MODELS=['lavie_base','magictime','cogvideox_5b','wanx21_1_3b','videocrafter2','opensora_plan_v1_3','ltx_video_095','pika_v2_2','cogvideox15_5b','latte','mochi1_preview',]
V_MAX3_MODELS=['anidiff','cogvideox_2b','hotshot_xl','opensora_v1_2','vchitect2',]
V_MAX2_MODELS=['ltx_video_091','text2video_zero','modelscope','zeroscope',]

P_MAX4_MODELS=['magictime','wanx21_1_3b','opensora_plan_v1_3','pika_v2_2','cogvideox15_5b','mochi1_preview','ltx_video_095',]
P_MAX3_MODELS=['anidiff','cogvideox_2b','hotshot_xl','cogvideox_5b','videocrafter2','latte','lavie_base','vchitect2',]
P_MAX2_MODELS=['ltx_video_091','text2video_zero','modelscope','zeroscope','opensora_v1_2',]

score_mapping={
    1:"1-Very Poor",
    2:"2-Relatively Poor",
    3:"3-Medium",
    4:"4-Relatively Good",
    5:"5-Very good",
}

def modify_score(t2v_model,v_score,t_score,p_score):
    v_score_new=v_score
    t_score_new=t_score
    p_score_new=p_score
    
    if t2v_model in V_MAX2_MODELS:
        v_score_new=min(2,v_score)
    if t2v_model in V_MAX3_MODELS:
        v_score_new=min(3,v_score)
    if t2v_model in V_MAX4_MODELS:
        v_score_new=min(4,v_score)
        
    if t2v_model in P_MAX2_MODELS:
        p_score_new=min(2,p_score)
    if t2v_model in P_MAX3_MODELS:
        p_score_new=min(3,p_score)
    if t2v_model in P_MAX4_MODELS:
        p_score_new=min(4,p_score)
    
    return v_score_new,t_score_new,p_score_new



   
def critical_modify_raw(paths,new_path,batch_name):
    data=[]
    for path in paths:
        with open(path,"r") as f:
            data.extend(json.load(f))
    new_data=[]
    v_scores=[]
    t_scores=[]
    p_scores=[]
    for idx, raw_item in tqdm(enumerate(data)):
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
        v_score=None
        t_score=None
        p_score=None
        for idx,label_dict in enumerate(raw_item["labels"]):
            label = label_dict["data"]["label"]
            value = str(label_dict["data"]["value"])
            if "视觉质量评分" in label:
                v_score = int(re.search(r"\d+", value).group())
                if t2v_model in V_MAX2_MODELS:
                    v_score=min(2,v_score)
                if t2v_model in V_MAX3_MODELS:
                    v_score=min(3,v_score)
                if t2v_model in V_MAX4_MODELS:
                    v_score=min(4,v_score)
                raw_item["labels"][idx]['data']['value']=score_mapping[v_score]
                v_scores.append(v_score)
                
            elif "文本符合度评分" in label:
                t_score = int(re.search(r"\d+", value).group())
                t_scores.append(t_score)
                
            elif "物理符合度评分" in label:
                p_score = int(re.search(r"\d+", value).group())
                if t2v_model in P_MAX2_MODELS:
                    p_score=min(2,p_score)
                if t2v_model in P_MAX3_MODELS:
                    p_score=min(3,p_score)
                if t2v_model in P_MAX4_MODELS:
                    p_score=min(4,p_score)
                raw_item["labels"][idx]['data']['value']=score_mapping[p_score] 
                p_scores.append(p_score)
                
            elif "视觉质量描述" in label:
                visual_cmt = value
                if visual_cmt==" ":
                    if v_score==5:
                        raw_item["labels"][idx]['data']['value']="visual quality is very good"
                    if v_score==4:
                        raw_item["labels"][idx]['data']['value']="v is 4"
                    # if v_score==3:
                    #     raw_item["labels"][idx]['data']['value']="v is 3"
                    # if v_score==2:
                    #     raw_item["labels"][idx]['data']['value']="v is 2"
                    if v_score==1:
                        raw_item["labels"][idx]['data']['value']="visual quality is very bad"
            elif "文本符合度描述" in label:
                t2v_cmt = value
                if t2v_cmt==" ":
                    if t_score==5:
                        raw_item["labels"][idx]['data']['value']="t2v_alignment is very good"
            elif "物理符合度描述" in label:
                phy_cmt = value
                if phy_cmt==" ":
                    if p_score==5:
                        raw_item["labels"][idx]['data']['value']="physical/common-sense consistency is very good"
                    # if p_score==4:
                    #     raw_item["labels"][idx]['data']['value']="p is 4"
                    # if p_score==3:
                    #     raw_item["labels"][idx]['data']['value']="p is 3"
                    # if p_score==2:
                    #     raw_item["labels"][idx]['data']['value']="p is 2"
                    # if p_score==1:
                    #     raw_item["labels"][idx]['data']['value']="p is 1"
        
        new_data.append(raw_item)

    plot(v_scores,batch_name,1)
    plot(t_scores,batch_name,2)
    plot(p_scores,batch_name,3)
    os.makedirs(os.path.dirname(new_path),exist_ok=True)
    with open(new_path,'w') as f:
        json.dump(new_data,f,indent=4,ensure_ascii=False)
    



def critical_modify_tk(path_or_plist,new_p,batch_name):
    data=[]
    if isinstance(path_or_plist,list):
        for path in path_or_plist:
            with open(path,"r") as f:
                data.extend(json.load(f))
    elif isinstance(path_or_plist,str):
        with open(path_or_plist,"r") as f:
            data=json.load(f)
    else:
        raise ValueError("type of path_or_plist must be str or list!")
    
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
     
    
 
    
    
if __name__ == "__main__":
    # batch_idx="39"
    # paths=[
    #     f"anno_raw/{batch_idx}.json"
    # ]
    # new_temp_path=f"temp/{batch_idx}_raw_modified.json"   
    # batch_name=f"{batch_idx}_raw_modified"     
    # critical_modify_raw(paths,new_temp_path,batch_name)

    original_dir="thinking_original"
    new_score_dir="thinking_new_score"
    batch_names=[fname.split("thinking_")[-1].split(".")[0] for fname in sorted(os.listdir(original_dir))]
    for batch_name in batch_names:
        path=f"{original_dir}/thinking_{batch_name}.json"
        new_path=f"{new_score_dir}/tk_new_score_{batch_name}.json"
        critical_modify_tk(path,new_path,batch_name)