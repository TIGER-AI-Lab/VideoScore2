import json
import os
from tqdm import tqdm
import re
import matplotlib.pyplot as plt


def plot(data,batch_name,dim_idx):
    if len(data)==0:
        return
    
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
    save_path=f"plots/{batch_name}_dim{dim_idx}.png"
    os.makedirs(os.path.dirname(save_path),exist_ok=True)
    plt.savefig(save_path)
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


def think_len_dist(paths,batch_name):
    think_len_list=[]
    for p in paths:
        with open(p,'r',encoding='utf-8') as f:
            data=json.load(f)
        for x in data:
            # if len(x['thinking'])<1000:
            #     print(p)
            #     print(x['video_name'])
            #     exit()
            if x['thinking'] is not None:
                think_len_list.append(len(x['thinking']))
            else:
                print(p)
                print(x['video_name'])
    
    bin_range=list(range(min(think_len_list)-100,max(think_len_list)+100,100))
    plt.hist(think_len_list, bins=bin_range, edgecolor='black', rwidth=0.8)
    plt.xlabel('Thinking Length')
    plt.ylabel('Frequency')
    plt.title(f'Batch {batch_name} Thinking Len Distribution')
    os.makedirs('plots_think_len',exist_ok=True)
    plt.savefig(f"plots_think_len/{batch_name}_think_len.png")
    plt.clf()
    
    # from transformers import AutoTokenizer
    # tokenizer = AutoTokenizer.from_pretrained("gpt2")
    # token_num_list=[]
    # for p in paths:
    #     with open(p,'r',encoding='utf-8') as f:
    #         data=json.load(f)
    #         for x in tqdm(data):
    #             token_num_list.append(len(tokenizer.encode(x['thinking'])))
    # bin_range=list(range(min(token_num_list)-20,max(token_num_list)+20,20))
    # plt.hist(token_num_list, bins=bin_range, edgecolor='black', rwidth=0.8)
    # plt.xlabel('Thinking Tokens Num')
    # plt.ylabel('Frequency')
    # plt.title(f'Batch {batch_name} Thinking Tokens Num Distribution')
    # os.makedirs('plots_think_len',exist_ok=True)
    # plt.savefig(f"plots_think_len/{batch_name}_think_token_num.png")
    # plt.clf()
    


if __name__ == "__main__":

    # anno_local_paths=[
    #     # 'anno_raw/37.json',
    #     # 'anno_raw/38.json',
    #     # 'anno_raw/39.json',
    #     # 'anno_raw/40.json',
    #     'anno_raw/10.json'
    # ]
    # batch_name="10_raw"
    # analyze_raw(anno_local_paths,batch_name)
    
    
    thinking_paths=[
        "thinking_new_score/tk_new_score_45.json",
    ]
    batch_name="45_modified"
    analyze_thinking(thinking_paths,batch_name)
    
    
    # batch_name="final_1_gpt-5-nano"
    # paths=[
    #     f"thinking_final/{batch_name}.json"
    # ]
    # think_len_dist(paths,batch_name)
    
    # batch_name="tk_new_score_1"
    # paths=[
    #     f"thinking_new_score/{batch_name}.json"
    # ]
    # think_len_dist(paths,batch_name)