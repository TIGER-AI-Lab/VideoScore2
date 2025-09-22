import json
import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

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
    for anno in tqdm(annos):
        video_name=anno['video_name']
        prompt_idx=video_name.split("_")[0]
        for item in prompt_items:
            if item['video_id']==prompt_idx:
                src=item['src']
                prompt_src_cnt[src]+=1
                break
            
    print(prompt_src_cnt)
     
    
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
                think_len_list.append(len(x['thinking']))
            else:
                print(p)
                print(x['video_name'])
    
    bin_range=list(range(min(think_len_list)-100,max(think_len_list)+100,100))
    plt.hist(think_len_list, bins=bin_range, rwidth=0.88)
    plt.xlabel('Thinking Length')
    plt.ylabel('Frequency')
    plt.title(f'Thinking Length Distribution')
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

    light_blue  = "#52C3F8"  # 淡蓝色
    light_green = "#90EE90"  # 淡绿色
    light_yellow= "#EEB24B"  # 淡黄色
    
    plt.bar(x - width, v_counts, width, color=light_blue, label="Visual Quality")
    plt.bar(x,         t_counts, width, color=light_green, label="Text Alignment")
    plt.bar(x + width, p_counts, width, color=light_yellow, label="Physical Consistency")

    plt.xticks(x, bins)
    plt.xlabel("Score")
    plt.ylabel("Count")
    plt.title("Score Distribution for All Dimensions")
    plt.legend()

    fig_dir = "paper_figures"
    os.makedirs(fig_dir, exist_ok=True)
    plt.savefig(f"{fig_dir}/score_dist_3_in_1.png")
    plt.clf()

if __name__ == "__main__":
    # prompt_sources()
    # prompt_word_cloud()
    prompt_len_hist()
    # think_len_dist()
    # score_dist_seperate()
    # score_dist_all_in_one()