import json
import os
from tqdm import tqdm
import re
import matplotlib.pyplot as plt


def plot(data,batch_name,dim_idx):
    dim={
        1:"Visual Quality",
        2:"T2V Alignment",
        3:"Phy Consistency"
    }
    plt.hist(data, bins=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5], edgecolor='black', rwidth=0.8)
    plt.xticks([1, 2, 3, 4, 5])
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.title(f'Batch {batch_name} {dim[dim_idx]} Score Distribution')
    plt.savefig(f"plots/{batch_name}_dim{dim_idx}.png")
    plt.clf()
    
    
def analyze_raw(anno_local_paths,batch_name):
    raw_annos=[]
    for anno_local_path in anno_local_paths:
        with open(anno_local_path,"r",encoding="utf-8") as f:
            raw_annos.extend(json.load(f))
    v_scores=[]
    t_scores=[]
    p_scores=[]
    
    for anno in tqdm(raw_annos):
        url=anno["info"]["data"][2]["content"]
        video_name=url.split("/")[-1].split(".")[0]
        try:
            visual_score=None
            t2v_score=None
            phy_score=None
            for label_dict in anno["labels"]:
                if "视觉质量评分" in label_dict["data"]["label"]:
                    visual_score=int(re.search(r'\d+', str(label_dict["data"]["value"])).group())
                if "文本符合度评分" in label_dict["data"]["label"]:
                    t2v_score=int(re.search(r'\d+', str(label_dict["data"]["value"])).group())
                if "物理符合度评分" in label_dict["data"]["label"]:
                    phy_score=int(re.search(r'\d+', str(label_dict["data"]["value"])).group())

            if visual_score is None:
                raise ValueError(f"visual score not found for {video_name}")
            if t2v_score is None:
                raise ValueError(f"t2v score not found for {video_name}")
            if phy_score is None:
                raise ValueError(f"phy score not found for {video_name}")
        except Exception as e:
            print(e)
            continue
        v_scores.append(visual_score)
        t_scores.append(t2v_score)
        p_scores.append(phy_score)
    
    
    plot(v_scores,batch_name,1)
    plot(t_scores,batch_name,2)
    plot(p_scores,batch_name,3)


def analyze_thinking(thinking_paths,batch_name):
    items=[]
    for p in thinking_paths:
        with open(p,"r",encoding="utf-8") as f:
            items.extend(json.load(f))
    v_scores=[]
    t_scores=[]
    p_scores=[]
    for item in tqdm(items):
        v_scores.append(item["visual_score"])
        t_scores.append(item["t2v_score"])
        p_scores.append(item["phy_score"])
        
    plot(v_scores,batch_name,1)
    plot(t_scores,batch_name,2)
    plot(p_scores,batch_name,3)
    
    


if __name__ == "__main__":

    # anno_local_paths=[
    #     # 'anno_raw/37.json',
    #     # 'anno_raw/38.json',
    #     # 'anno_raw/39.json',
    #     # 'anno_raw/40.json',
    #     'anno_raw/54.json'
    # ]
    # batch_name="54_keming"
    # analyze_raw(anno_local_paths,batch_name)
    
    
    thinking_paths=[
        "temp/sft_17k_modidifed.json",
    ]
    batch_name="17k_modified"
    analyze_thinking(thinking_paths,batch_name)