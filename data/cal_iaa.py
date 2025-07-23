import os
import json
import re
from scipy.stats import spearmanr, pearsonr
import numpy as np


score_map={
    "5-Very Good":5,
    "4-Relatively Good":4,
    "3-Medium":3,
    "2-Relatively Poor":2,
    "1-Very Poor":1,
    "Very Good":5,
    "Relatively Good":4,
    "Medium":3,
    "Relatively Poor":2,
    "Very Poor":1,
    "5":5,
    "4":4,
    "3":3,
    "2":2,
    "1":1,
}


def three_way_agreement_ratio(a, b, c):
    assert len(a) == len(b) == len(c)
    agree_count = sum(1 for x, y, z in zip(a, b, c) if x == y == z)
    return agree_count / len(a)

def three_way_spcc(a, b, c):
    data = np.array([a, b, c])
    corr_matrix, _ = spearmanr(data, axis=1)
    # 只取非对角的三个相关值，求平均
    spcc_avg = (corr_matrix[0,1] + corr_matrix[0,2] + corr_matrix[1,2]) / 3
    return spcc_avg


def import_anno(p):
    # for p in ["try1.json","try2.json","try3.json"]:
    
    with open(p,"r",encoding='utf-8') as f:
        annos=json.load(f)
    
    annos=[item for item in annos if len(item["labels"])]
    
    data=[]
    
    for anno in annos:
        prompt_en = (
            anno["info"]["data"][1]["content"]
            .split("English Prompt", 1)[1]
            .split("\n", 1)[0]
            .strip(". :\n")
        )
        
        url = anno["info"]["data"][2]["content"]
        video_name=url.split("/")[-1].split('.')[0]

        for label_dict in anno["labels"]:
            label = label_dict["data"]["label"]
            if not ("视觉质量评分" in label or "文本符合度评分" in label or "物理符合度评分" in label):
                continue
            value = str(label_dict["data"]["value"])

            if "视觉质量评分" in label:
                visual_score = score_map[value]
            elif "文本符合度评分" in label:
                t2v_score = score_map[value]
            elif "物理符合度评分" in label:
                phy_score = score_map[value]
        
        data.append({
            'video_name':video_name,
            'video_url':url,
            'prompt':prompt_en,
            'visual_score':visual_score,
            't2v_score':t2v_score,
            'phy_score':phy_score
        })
    print(len(data))
    return data


def cal_iaa(data1,data2,data3):
    video_names_1=[x['video_name'] for x in data1]
    video_names_2=[x['video_name'] for x in data2]
    video_names_3=[x['video_name'] for x in data3]
    
    shared=list(set(video_names_1)&set(video_names_2)&set(video_names_3))
    
    data1=[x for x in data1 if x['video_name'] in shared]
    data2=[x for x in data2 if x['video_name'] in shared]
    data3=[x for x in data3 if x['video_name'] in shared]
    
    v_scores_2dlist=[
        [x['visual_score'] for x in data1],
        [x['visual_score'] for x in data2],
        [x['visual_score'] for x in data3],
    ]
    
    t_scores_2dlist=[
        [x['t2v_score'] for x in data1],
        [x['t2v_score'] for x in data2],
        [x['t2v_score'] for x in data3],
    ]
    
    p_scores_2dlist=[
        [x['phy_score'] for x in data1],
        [x['phy_score'] for x in data2],
        [x['phy_score'] for x in data3],
    ]
    
    for _2dlist in [v_scores_2dlist,t_scores_2dlist,p_scores_2dlist]:
        print(f"  Agreement Ratio: {three_way_agreement_ratio(_2dlist[0],_2dlist[1],_2dlist[2],):.4f}")
        print(f"  SPCC: {three_way_spcc(_2dlist[0],_2dlist[1],_2dlist[2],):.4f}")
        print()

    
if __name__ == "__main__":
    path="try1.json"
    data1=import_anno(path)
    path="try2.json"
    data2=import_anno(path)
    path="try3.json"
    data3=import_anno(path)
    
    cal_iaa(data1,data2,data3)
    
    