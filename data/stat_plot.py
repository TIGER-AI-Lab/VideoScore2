import json
import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import copy
def prompt_sources():
    p="/data/xuan/data/videoscore2/text_prompts/all_prompts.jsonl"
    with open(p, "r", encoding="utf-8") as f:
        prompt_items = [json.loads(line) for line in f]
    
    paths=[
        f"thinking_final/{fname}" for fname in os.listdir("thinking_final") if fname.endswith('.json')
    ]
    annos=[]
    for path in paths:
        annos.extend(json.load(open(path,"r",encoding='utf-8')))
    prompt_src_cnt={
        "vidprom":0,
        "koala":0,
        "ocr_text":0,
        "story":0,
        "camera_motion":0,
    }
    total=len(annos)
    for anno in tqdm(annos):
        video_name=anno['video_name']
        prompt_idx=video_name.split("_")[0]
        for item in prompt_items:
            if item['video_id']==prompt_idx:
                src=item['src']
                prompt_src_cnt[src]+=1
                break
            
    print(prompt_src_cnt)
    for k in prompt_src_cnt:
        print(f"{k}: {prompt_src_cnt[k]}/{total}, {prompt_src_cnt[k]/total:.2%}")
    
def prompt_word_cloud():
    from wordcloud import WordCloud
    import random
    p="/data/xuan/data/videoscore2/text_prompts/all_prompts.jsonl"
    with open(p, "r", encoding="utf-8") as f:
        prompt_items = [json.loads(line) for line in f]
    random.seed(42)
    random.shuffle(prompt_items)
    prompt_items=random.sample(prompt_items, 200)
    
    text=" ".join([item['text'] for item in prompt_items])
    trigger_words=["camera"]
    for word in trigger_words:
        text=text.replace(word, "")
    text=text.replace("\n", " ")
    print(len(text))
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    fig_dir="paper_figures"
    os.makedirs(fig_dir, exist_ok=True)
    plt.savefig(f"{fig_dir}/prompt_wordcloud.png")
    plt.clf()
    
    
def prompt_len_hist():
    p="/data/xuan/data/videoscore2/text_prompts/all_prompts.jsonl"
    with open(p, "r", encoding="utf-8") as f:
        prompt_items = [json.loads(line) for line in f]
    lens=[len(item['text'].split()) for item in prompt_items]
    print(np.mean(lens), np.median(lens), np.max(lens), np.min(lens))
    plt.hist(lens, bins=30,rwidth=0.88,)
    plt.xlabel("Prompt Length (words)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Prompt Lengths")
    fig_dir="paper_figures"
    os.makedirs(fig_dir, exist_ok=True)
    plt.savefig(f"{fig_dir}/prompt_len_hist.png")
    plt.clf()


def think_len_dist():
    batch_name=""
    paths=[
        f"thinking_final/{fname}" for fname in os.listdir("thinking_final") if fname.endswith('.json')
    ]
    think_len_list=[]
    for p in paths:
        with open(p,'r',encoding='utf-8') as f:
            data=json.load(f)
        for x in data:
            if x['thinking'] is not None:
                think_len_list.append(len(x['thinking'].split(" ")))
            else:
                print(p)
                print(x['video_name'])
    plt.figure(figsize=(8, 4))
    color1  = "#68E3F4FB"  
    bin_range=list(range(min(think_len_list)-50,max(think_len_list)+50,20))
    plt.hist(think_len_list, bins=bin_range, rwidth=0.88, color=color1)
    plt.xlabel('Num of words in rationale (thinking)')
    plt.ylabel('Frequency')
    plt.title(f'Rationale Length Distribution')
    fig_dir="paper_figures"
    os.makedirs(fig_dir, exist_ok=True)
    plt.savefig(f"{fig_dir}/{batch_name}_think_len.png")
    plt.clf()

def score_dist_seperate():
    paths=[
        f"thinking_final/{fname}" for fname in os.listdir("thinking_final") if fname.endswith('.json')
    ]
    annos=[]
    for path in paths:
        annos.extend(json.load(open(path,"r",encoding='utf-8')))
        
    v_scores=[]
    t_scores=[]
    p_scores=[]
    for anno in annos:
        v_scores.append(anno['visual_score'])
        t_scores.append(anno['t2v_score'])
        p_scores.append(anno['phy_score'])
    
    def plot(scores,score_type):
        plt.hist(scores, bins=[1,2,3,4,5,6], rwidth=0.88, align='left', density=True)
        plt.xticks([1,2,3,4,5])
        plt.xlabel(f'{["Visual","T2V","Physical"][score_type-1]} Score')
        plt.ylabel('Frequency')
        plt.title(f'{["Visual","T2V","Physical"][score_type-1]} Score Distribution')
        fig_dir="paper_figures"
        os.makedirs(fig_dir, exist_ok=True)
        plt.savefig(f"{fig_dir}/score_dist_{['visual','t2v','phy'][score_type-1]}.png")
        plt.clf()
    
    plot(v_scores,1)
    plot(t_scores,2)
    plot(p_scores,3)


def score_dist_all_in_one():
    paths = [
        f"thinking_final/{fname}" for fname in os.listdir("thinking_final") if fname.endswith('.json')
    ]
    annos = []
    for path in paths:
        annos.extend(json.load(open(path, "r", encoding="utf-8")))

    v_scores = [anno["visual_score"] for anno in annos]
    t_scores = [anno["t2v_score"] for anno in annos]
    p_scores = [anno["phy_score"] for anno in annos]

    # 统计每个分数的频率
    bins = [1, 2, 3, 4, 5]
    v_counts = [v_scores.count(b) for b in bins]
    t_counts = [t_scores.count(b) for b in bins]
    p_counts = [p_scores.count(b) for b in bins]

    x = np.arange(len(bins))  # 分数位置
    width = 0.25              # 每根柱子的宽度

    color1  = "#82F2C5"  
    color2 = "#AFC8EB"  
    color3= "#EAB3A7"  
    plt.figure(figsize=(8, 4))
    
    plt.bar(x - width, v_counts, width, color=color1, label="Visual Quality")
    plt.bar(x,         t_counts, width, color=color2, label="Text Alignment")
    plt.bar(x + width, p_counts, width, color=color3, label="Physical Consistency")

    plt.xticks(x, bins)
    plt.xlabel("Score")
    plt.ylabel("Count")
    plt.title("Score Distribution in Annotations")
    plt.legend()

    fig_dir = "paper_figures"
    os.makedirs(fig_dir, exist_ok=True)
    plt.savefig(f"{fig_dir}/score_dist_3_in_1.png")
    plt.clf()


def count_score_difference_dist():
    
    paths=[
        f"/data/xuan/workdir/VideoScore2/data/thinking_original/thinking_{bname}.json" for bname in [
            "com_5k_0",
            "com_5k_1",
            "com_5k_2",
            "com_5k_3",
            "com_5k_4",
        ]
        ]
    data=[]
    for p in paths:
        data.extend(json.load(open(p,"r",encoding='utf-8')))
    total_items=len(data)
    human_score=[]
    v_gt=[]
    t_gt=[]
    p_gt=[]    
    model_score=[]
    v_model=[]
    t_model=[]
    p_model=[]
    for item in tqdm(data):
        human_score.extend([item['visual_score'],item['t2v_score'],item['phy_score']])
        v_gt.append(item['visual_score'])
        t_gt.append(item['t2v_score'])
        p_gt.append(item['phy_score'])
        model_score.extend([item['visual_score_model'],item['t2v_score_model'],item['phy_score_model']])
        v_model.append(item['visual_score_model'])
        t_model.append(item['t2v_score_model'])
        p_model.append(item['phy_score_model'])

    diff_dict={
        0:0,
        1:0,
        -1:0,   
        2:0,
        -2:0,
        3:0,
        -3:0,
        4:0,
        -4:0,
    }

    v_diff_dict=copy.deepcopy(diff_dict)
    t_diff_dict=copy.deepcopy(diff_dict)
    p_diff_dict=copy.deepcopy(diff_dict)
    
    for h,m in zip(human_score,model_score):
        diff=h-m
        diff_dict[diff]+=1
    for h,m in zip(v_gt,v_model):
        diff=h-m
        v_diff_dict[diff]+=1
    for h,m in zip(t_gt,t_model):
        diff=h-m
        t_diff_dict[diff]+=1
    for h,m in zip(p_gt,p_model):
        diff=h-m
        p_diff_dict[diff]+=1
    
    # print("V Score Difference Distribution:")
    # for k in sorted(v_diff_dict.keys()):
    #     print(f"{k}: {v_diff_dict[k]}/{total_items}")
    
    # print("T Score Difference Distribution:")
    # for k in sorted(t_diff_dict.keys()):
    #     print(f"{k}: {t_diff_dict[k]}/{total_items}")
        
    # print("P Score Difference Distribution:")
    # for k in sorted(p_diff_dict.keys()):
    #     print(f"{k}: {p_diff_dict[k]}/{total_items}")
        
    print("Overall Score Difference Distribution:")
    for k in sorted(diff_dict.keys()):
        print(f"{k}: {diff_dict[k]}/{total_items*3}")
    
if __name__ == "__main__":
    prompt_sources()
    # prompt_word_cloud()
    # prompt_len_hist()
    # think_len_dist()
    # score_dist_seperate()
    # score_dist_all_in_one()
    # count_score_difference_dist()
    
