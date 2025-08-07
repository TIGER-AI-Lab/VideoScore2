
ROUND_DIGIT=3

def plot(data,batch_name,dim_idx):
    import matplotlib.pyplot as plt
    import os
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
    os.makedirs("plots",exist_ok=True)
    plt.savefig(f"plots/{batch_name}_dim{dim_idx}.png")
    plt.clf()


def plot_float(data,batch_name,dim_idx):
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    bin_width = 0.01
    min_val = np.floor(min(data))
    max_val = np.ceil(max(data))
    bins = np.arange(min_val, max_val + bin_width, bin_width)
    counts, bin_edges, _ = plt.hist(data, bins=bins, edgecolor='black')

    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.title("Histogram with Bin Size 0.5")
    plt.grid(True)
    plt.title(f'{batch_name} dim{dim_idx} Score Distribution')
    os.makedirs("plots",exist_ok=True)
    plt.savefig(f"plots/{batch_name}_dim{dim_idx}.png")
    plt.clf()


def load_scores(score_res_path):
    import json
    import re
    with open(score_res_path,"r") as f:
        data=json.load(f)
    
    v_scores_gt=[x['v_score_gt'] for x in data]
    t_scores_gt=[x['t_score_gt'] for x in data]
    p_scores_gt=[x['p_score_gt'] for x in data]
    v_scores_model=t_scores_model=p_scores_model=[]
    
    if all(f"{dim}_score_model" in x for dim in ["v","t","p"] for x in data) :
        v_scores_model=[x['v_score_model'] for x in data]
        t_scores_model=[x['t_score_model'] for x in data]
        p_scores_model=[x['p_score_model'] for x in data]
    
    else:
        pattern = r"visual quality:\s*(\d+).*?text-to-video alignment:\s*(\d+).*?physical/common-sense consistency:\s*(\d+)"
        for x in data:
            video_name=x['video_name']
            output=x['output'][-150:]
            try:
                match = re.search(pattern, output, re.DOTALL | re.IGNORECASE)
                if match:
                    v_scores_model.append(max(int(match.group(1)),1))
                    t_scores_model.append(max(int(match.group(2)),1))
                    p_scores_model.append(max(int(match.group(3)),1))
            
                else:
                    print(f"{video_name} no matched score")
                    v_scores_model.append(0)
                    t_scores_model.append(0)
                    p_scores_model.append(0)

            except Exception as e:
                print(f'[err] {e}')
                print(f"{video_name} no matched score")
                v_scores_model.append(0)
                t_scores_model.append(0)
                p_scores_model.append(0)
    
    return v_scores_gt, t_scores_gt, p_scores_gt, v_scores_model, t_scores_model, p_scores_model
    
    
def remove_null_scores(v_scores_gt, t_scores_gt, p_scores_gt, v_scores_model, t_scores_model, p_scores_model):
    new_v_gt = new_t_gt = new_p_gt = new_v_model = new_t_model = new_p_model = []
    for v_g, t_g, p_g, v_m, t_m, p_m in zip(v_scores_gt, t_scores_gt, p_scores_gt, v_scores_model, t_scores_model, p_scores_model):
        if None not in (v_g, t_g, p_g, v_m, t_m, p_m):
            new_v_gt.append(v_g)
            new_t_gt.append(t_g)
            new_p_gt.append(p_g)
            new_v_model.append(v_m)
            new_t_model.append(t_m)
            new_p_model.append(p_m)

    return new_v_gt, new_t_gt, new_p_gt, new_v_model, new_t_model, new_p_model

def compute_accuracy(pred, ground_truth):
    assert len(pred) == len(ground_truth), "len(pred) should be the same as len(ground_truth)"
    try:
        pred = [0 if x is None else x for x in pred]
        ground_truth  = [-1 if gt is None else gt for gt in ground_truth]
        correct = sum(p == gt for p, gt in zip(pred, ground_truth))
        total = len(ground_truth)
        return round(correct / total*100,ROUND_DIGIT) if total > 0 else 0.0
    
    except Exception as e:
        print(e)
        return None


def compute_accuracy_relaxed(pred, ground_truth):
    assert len(pred) == len(ground_truth), "len(pred) should be the same as len(ground_truth)"
    try:
        pred = [0 if x is None else x for x in pred]
        ground_truth  = [-1 if gt is None else gt for gt in ground_truth]
        correct = sum(abs(p-gt)<=1 for p, gt in zip(pred, ground_truth))
        total = len(ground_truth)
        
        return round(correct / total *100,ROUND_DIGIT) if total > 0 else 0.0
    except Exception as e:
        print(e)
        return None

def acc_whole_item(pred1,pred2,pred3,gt1,gt2,gt3):
    matched=0
    total = len(gt1)
    assert len(pred1) == len(gt1) and len(pred2) == len(gt2) and len(pred3) == len(gt3), "len(pred) should be the same as len(ground_truth)"
    try:
        for p1, p2, p3, g1, g2, g3 in zip(pred1, pred2, pred3, gt1, gt2, gt3):
            if p1 is None or p2 is None or p3 is None:
                continue
            if g1 is None or g2 is None or g3 is None:
                continue
            if abs(p1 - g1) + abs(p2 - g2) + abs(p3 - g3) <= 0:
                matched += 1
        return round(matched / total*100,ROUND_DIGIT) if total > 0 else 0.0
    except Exception as e:
        print(e)
        return None
        
    
def compute_spcc(pred, ground_truth):
    try:
        new_pred = []
        new_gt = []
        for ai, bi in zip(pred, ground_truth):
            if ai is not None and bi is not None:
                new_pred.append(ai)
                new_gt.append(bi)
        
        from scipy.stats import spearmanr
        assert len(new_pred) == len(new_gt), "len(pred) should be the same as len(ground_truth)"
        coefficient, _ = spearmanr(new_pred, new_gt)
        coefficient=float(coefficient)
        return round(coefficient *100 ,ROUND_DIGIT)
    except Exception as e:
        print(e)
        return None


def compute_plcc(pred, ground_truth):
    try:
        new_pred = []
        new_gt = []
        for ai, bi in zip(pred, ground_truth):
            if ai is not None and bi is not None:
                new_pred.append(ai)
                new_gt.append(bi)
        
        from scipy.stats import pearsonr
        assert len(new_pred) == len(new_gt), "len(pred) should be the same as len(ground_truth)"
        coefficient, _ = pearsonr(new_pred, new_gt)
        coefficient=float(coefficient)
        return round(coefficient*100, ROUND_DIGIT)
    except Exception as e:
        print(e)
        return None

    
    
def get_acc(method_name,bench_name,score_res_path,metric_report_p):
    if bench_name in ["genai_bench","videogen_reward_bench","vision_reward_db_video"]:
        raise ValueError(f"{bench_name} is a preference benchmark, SPCC/PLCC is not supported")
    
    v_scores_gt, t_scores_gt, p_scores_gt, v_scores_model, t_scores_model, p_scores_model \
        = load_scores(score_res_path)
    
    overall_scores_gt=[]
    overall_scores_model=[]
    
    # To calculate Accuracy, rescale for different reward models / eval methods
    if method_name in ["vs2"]:
        None
    
    if method_name in ["unified_reward"]:
        # UnifiedReward (the version for video generation point score) has 1 dim (broadcast to 3).
        # Raw score (float) in [1,4]. Rescale to [1,2,3,4,5].      
        v_scores_model = t_scores_model = p_scores_model =[
            int(round(x*1.25)) for x in v_scores_model
        ]
        
    if method_name in ["vision_reward"]:
        # VisionReward has 1 dim (final score), broadcast to 3 dim. 
        # Raw score (float) in [-0.25, 0.25]. Assume Gaussian Dist(0, 0.2). Rescale to [1,2,3,4,5]  
        from scipy.stats import norm  
        v_scores_model = t_scores_model = p_scores_model =\
        [
            1 if z/0.2 < norm.ppf(0.2) else 
            2 if z/0.2 < norm.ppf(0.4) else 
            3 if z/0.2 < norm.ppf(0.6) else 
            4 if z/0.2 < norm.ppf(0.8) else 5
            for z in v_scores_model
        ]
        
    if method_name in ["video_reward"]:
        # VideoReward has 2 dim (v t), v [-2,2], t [-3,3]. 
        # Raw score (float). Assume Gaussian Dist. Rescale to [1,2,3,4,5]  
        from scipy.stats import norm 
        v_scores_model = [
            1 if z < norm.ppf(0.2) else 
            2 if z < norm.ppf(0.4) else 
            3 if z < norm.ppf(0.6) else 
            4 if z < norm.ppf(0.8) else 5
            for z in v_scores_model
        ]
        t_scores_model = [
            1 if z/1.5 < norm.ppf(0.2) else 
            2 if z/1.5 < norm.ppf(0.4) else 
            3 if z/1.5 < norm.ppf(0.6) else 
            4 if z/1.5 < norm.ppf(0.8) else 5
            for z in t_scores_model
        ]
        p_scores_model = [-1 for x in p_scores_model]        
    
    if method_name in ["image_reward"]:
        # ImageReward has 1 dim (final score), broadcast to 3 dim. 
        # Raw score (float). Normalized to have mean=1 and std=1. Assume Gaussian Dist. Rescale to [1,2,3,4,5]  
        from scipy.stats import norm 
        v_scores_model = t_scores_model = p_scores_model = [
            1 if z < norm.ppf(0.2) else 2 if z < norm.ppf(0.4) else 3 if z < norm.ppf(0.6) else 4 if z < norm.ppf(0.8) else 5
            for z in v_scores_model
        ]
    
    if method_name in ["aigve_macs"]:
        None
        
    if method_name in ["lift"]:
        # LiFT has 3 dim {fiedlity, semantic, motion}, dim1 and dim2 are matched with vs2. 
        # Raw score (int) in [1,2,3]. 
        if bench_name in ["aigve_bench","mj_bench_video","video_phy2"]:
            # the above benchmarks have <=3 score levels, which LiFT score can be adapted.
            # For other benchmarks, skip LiFT for acc calculation
            v_scores_model=[1 if x==1 else 3 if x==2 else 5 for x in v_scores_model]
            t_scores_model=[1 if x==1 else 3 if x==2 else 5 for x in t_scores_model]
            p_scores_model=[-1 for x in p_scores_model]
        else:
            print("[skip] skipping acc calculation for method LiFT. Exited")      
            return None
        
    if method_name in ["video_phy2_auto_eval"]:
        # VideoPhy2-AutoEval has 2 dim (t p). Raw score (int) in [1,2,3,4,5]  
        v_scores_model = [-1 for x in v_scores_model]    
        
    if method_name in ["dover"]:
        # DOVER is used for VQ, broadcast to 3 dim. 
        # Raw score (float) in [0,1]. Rescale to [1,2,3,4,5] 
        v_scores_model = t_scores_model = p_scores_model =  [min(5, max(1, round(5*x))) for x in v_scores_model]  
    
    if method_name in ["q_insight"]:
        # Q-Insight can predict 3 dims, v t p.
        # Raw score (float) in [0,5]. Rescale to [1,2,3,4,5]
        v_scores_model = [min(5, max(1, round(x))) for x in v_scores_model]
        t_scores_model = [min(5, max(1, round(x))) for x in t_scores_model]
        p_scores_model = [min(5, max(1, round(x))) for x in p_scores_model]
    
    if method_name in ["q_align"]:
        # Q-Align has 1 dim (final score), broadcast to 3 dim. 
        # Raw score (float) in [0,1]. Rescale to [1,2,3,4,5].
        v_scores_model = t_scores_model = p_scores_model = [min(5, max(1, round(5*x))) for x in v_scores_model]
    
    if method_name in ["deqa"]:
        # DeQA has 1 dim (final score), broadcast to 3 dim. 
        # Raw score (float) in [0,5]. Rescale to [1,2,3,4,5].
        v_scores_model = t_scores_model = p_scores_model = [min(5, max(1, round(x))) for x in v_scores_model]

    
    
    # To calculate Accuracy, rescale for different benchmarks
    if "vs2" in bench_name:
        None
    
    if bench_name in ["aigve_bench",]:
        # In AIGVE-Bench, phy dim only has score 1,3,5
        # (1,2)->1, (3,4)->3, 5->5
        p_scores_model=[1 if x == 2 else 3 if x == 4 else x for x in p_scores_model]            
    
    if bench_name in ["video_phy_test",]:
        # In Video-Phy-test, sa and pc dim only have score 0,1
        # (1,2,3)->0, (4,5)->1
        t_scores_model = [0 if x in [1, 2, 3] else 1 for x in t_scores_model]
        p_scores_model = [0 if x in [1, 2, 3] else 1 for x in p_scores_model]
    
    if bench_name in ["video_phy2_test",]:
        None
        
    if bench_name in ["mj_bench_video",]:
        # In MJ-Bench-Video, v and t dim have score 0,1,2
        # (1,2)->0, (3,4)->1, 5->2
        v_scores_model = [0 if x in [1,2] else 1 if x in [3,4] else 2 for x in v_scores_model]
        t_scores_model = [0 if x in [1] else 1 if x in [2, 3] else 2 for x in t_scores_model]
        import json
        with open(score_res_path,"r") as f:
            tmp_data=json.load(f)
        overall_scores_gt=[x["total_score"] for x in tmp_data]
        overall_scores_model=[int((x["v_score_out"]+x["t_score_out"]+x["p_score_out"])/3) for x in tmp_data]
        overall_scores_model=[0 if x in [1] else 1 if x in [2, 3] else 2 for x in overall_scores_model]
        
    if bench_name in ["tvge","t2v_gen_eval"]:
        # In TVGE (or T2V Gen Eval), 2 dim v t. raw score is float in [1.0, 5.0].
        # Skip acc calculation for this benchmark.
        print("[skip] skipping acc calculation for benchmark TVGE. Exited")    
        return None
    
    if bench_name in ["t2vqa_db"]:
        # In T2VQA-DB, only one final score. raw score is float in [0,100].
        # Skip acc calculation for this benchmark.
        print("[skip] skipping acc calculation for benchmark T2VQA-DB. Exited")    
        return None
        
        
    metrics_dict={
        "v_acc":compute_accuracy(v_scores_model,v_scores_gt),
        "t_acc":compute_accuracy(t_scores_model,t_scores_gt),
        "p_acc":compute_accuracy(p_scores_model,p_scores_gt),
        
        "v_acc_relaxed":compute_accuracy_relaxed(v_scores_model,v_scores_gt),
        "t_acc_relaxed":compute_accuracy_relaxed(t_scores_model,t_scores_gt),
        "p_acc_relaxed":compute_accuracy_relaxed(p_scores_model,p_scores_gt),
        
        "overall_acc":compute_accuracy(overall_scores_model,overall_scores_gt),
        
        "acc_whole_item":acc_whole_item(v_scores_model,t_scores_model,p_scores_model,v_scores_gt,t_scores_gt,p_scores_gt),
        }
    print(list(metrics_dict.items())[:3])
    print(list(metrics_dict.items())[3:6])
    print(list(metrics_dict.items())[6:7])
    print(list(metrics_dict.items())[7:])
    
    # batch_name="sft_model_score" 
    # plot(v_scores_model,batch_name,1)
    # plot(t_scores_model,batch_name,2)
    # plot(p_scores_model,batch_name,3)
    
    # with open(metric_report_p,"w") as f:
    #     json.dump({
    #         method_name:metrics_dict
    #     },f,indent=4)
    
    
    
    
    
def get_corr(method_name,bench_name,score_res_path,metric_report_p):
    if bench_name in ["genai_bench","videogen_reward_bench","vision_reward_db_video"]:
        raise ValueError(f"{bench_name} is a preference benchmark, SPCC/PLCC is not supported")
    
    v_scores_gt, t_scores_gt, p_scores_gt, v_scores_model, t_scores_model, p_scores_model \
        = load_scores(score_res_path)
    
    overall_scores_gt=[]
    overall_scores_model=[]
    
    # rescale for different **reward models / eval methods**
    if method_name in ["vs2","vs1",
                       "vision_reward","unified_reward","image_reward",
                       "aigve_macs","dover",
                       "q_insight","q_align","deqa"]:
        # use raw score to calculate SPCC/PLCC
        None
        
    if method_name in ["lift",]:
        # LiFT has 3 dim {fiedlity, semantic, motion}, dim1 and dim2 are matched with vs2. 
        # Raw score (int) in [1,2,3]. Convert gt for SPCC/PLCC calculation.
        v_scores_gt=[1 if x in [1,2] else 2 if x in [3] else 3 for x in v_scores_gt]
        t_scores_gt=[1 if x in [1,2] else 2 if x in [3] else 3 for x in t_scores_gt]
        p_scores_gt=[1 if x in [1,2] else 2 if x in [3] else 3 for x in p_scores_gt]
        
        p_scores_model=[None for x in p_scores_gt]

    if method_name in ["video_reward"]:
        # VideoPhy2-AutoEval has 2 dim (v t), ignore dim3
        # use raw VideoReward score to calculate SPCC/PLCC
        import warnings
        warnings.warn("For VideoReward, SPCC/PLCC for dim3 - p_score should be ignored.")
        
    if method_name in ["video_phy2_auto_eval"]:
        # VideoPhy2-AutoEval has 2 dim (t p), ignore dim1
        # use raw VideoPhy2-AutoEval score to calculate SPCC/PLCC
        v_scores_model = [None for x in v_scores_model]

    # rescale for different **benchmarks**
    if "vs2" in bench_name:
        None
    
    if bench_name in ["aigve_bench",]:
        None
    
    if bench_name in ["tvge",]:
        p_scores_model = [None for x in p_scores_model]
        p_scores_gt = [None for x in p_scores_gt]
    
    if bench_name in ["video_phy_test","video_phy2_test"]:
        v_scores_model = [None for x in v_scores_model]
        v_scores_gt = [None for x in v_scores_gt]
        
    if bench_name in ["t2vqa_db"]:
        overall_scores_model=[x+y+z for x,y,z in zip(v_scores_model,t_scores_model,p_scores_model)]
        overall_scores_gt=[x+y+z for x,y,z in zip(v_scores_gt,t_scores_gt,p_scores_gt)]
    
    if bench_name in ["mj_bench_video"]:
        import json
        with open(score_res_path,"r") as f:
            tmp_data=json.load(f)
        overall_scores_gt=[x["total_score"] for x in tmp_data]
        overall_scores_model=[int((x["v_score_out"]+x["t_score_out"]+x["p_score_out"])/3) for x in tmp_data]
    
    metrics_dict={        
        "v_spcc":compute_spcc(v_scores_model,v_scores_gt),
        "t_spcc":compute_spcc(t_scores_model,t_scores_gt),
        "p_spcc":compute_spcc(p_scores_model,p_scores_gt),
        "overall_spcc":compute_spcc(overall_scores_model,overall_scores_gt),
            
        "v_plcc":compute_plcc(v_scores_model,v_scores_gt),
        "t_plcc":compute_plcc(t_scores_model,t_scores_gt),
        "p_plcc":compute_plcc(p_scores_model,p_scores_gt),
        "overall_plcc":compute_plcc(overall_scores_model,overall_scores_gt),
         }
    print(list(metrics_dict.items())[:3])
    print(list(metrics_dict.items())[3:4])
    print(list(metrics_dict.items())[4:7])
    print(list(metrics_dict.items())[7:])
    
    
    # batch_name="sft_model_score" 
    # plot(v_scores_model,batch_name,1)
    # plot(t_scores_model,batch_name,2)
    # plot(p_scores_model,batch_name,3)
    
    # with open(metric_report_p,"w") as f:
    #     json.dump({
    #         method_name:metrics_dict
    #     },f,indent=4)
        
if __name__ == "__main__":
    # res_p="res_data/res_vs2_test_sft_17k/open-router-claude-sonnet-4.json"
    # res_p="res_data/res_vs2_test_sft_17k/open-router-claude-sonnet-4_infer_4fps.json"
    # res_p="res_data/res_vs2_test_sft_17k/open-router-gemini-2.5-flash.json"
    # res_p="res_data/res_vs2_test_sft_17k/open-router-gemini-2.5-flash_infer_4fps.json"
    # res_p="res_data/res_vs2_test_sft_17k/open-router-gemini-2.5-pro.json"
    # res_p="res_data/res_vs2_test_sft_17k/open-router-gpt-4.1_infer_4fps.json"
    # res_p="res_data/res_vs2_test_sft_17k/open-router-grok-4_infer_4fps.json"
    # res_p="res_data/res_vs2_test_sft_17k/open-router-o4-mini_infer_4fps.json"
    # res_p="res_data/res_vs2_test_sft_17k/open-router-gemma-3-27b-it_infer_4fps.json"
    # res_p="res_data/res_vs2_test_sft_17k/open-router-llama-4-scout_infer_4fps.json"
    # res_p="res_data/res_vs2_test_sft_17k/open-router-llama-4-maverick_infer_4fps.json"
    # res_p="res_data/res_vs2_test_sft_17k/open-router-qwen2.5-vl-32b-instruct_infer_4fps.json"
    # res_p="res_data/res_vs2_test_sft_17k/open-router-qwen2.5-vl-72b-instruct_infer_4fps.json"
    # res_p="res_data/res_vs2_test_sft_17k/open-router-glm-4.1v-9b-thinking_infer_4fps.json"
    # res_p="res_data/res_vs2_test_sft_17k/vs2_qwen2_5vl_sft_17k_2e-4_8fps_16384_infer_8fps.json"
    # res_p="res_data/res_vs2_test_sft_17k/vs2_qwen2_5vl_sft_17k_2e-4_2fps_512_512_8192_infer_8fps.json"
    # res_p="res_data/res_vs2_test_sft_17k/VideoScore.json"
    # res_p="res_data/res_vs2_test_sft_17k/feat_dino_sim.json"
    # res_p="res_data/res_vs2_test_sft_17k/VisionReward-Video.json"
    # res_p="res_data/res_vs2_test_sft_17k/VideoReward.json"
    # res_p="res_data/res_vs2_test_sft_17k/videophy_2_auto.json"
    # res_p="res_data/res_vs2_test_sft_17k/AIGVE-MACS.json"
    # res_p="res_data/res_vs2_test_sft_17k/vs2_qwen2_5vl_grpo_17k_try_1e-6_800_infer_4fps.json"
    # res_p="res_data/res_vs2_test_sft_17k/vs2_qwen2_5vl_grpo_17k_1e-6_reward_3_3200_infer_2fps.json"
    # res_p="res_data/res_vs2_test_sft_17k/vs2_qwen2_5vl_grpo_17k_1e-6_base960-720_reward_3_3000_infer_2fps.json"
    res_p="res_data/res_vs2_test_sft_17k/vs2_qwen2_5vl_sft_17k_2e-4_2fps_960_720_8192_infer_4fps.json"
    
    method_name="vs2"
    bench_name="vs2_test_sft_17k"

    # res_p="res_data/res_vs2_test_sft_17k/VisionReward-Video.json"
    # method_name="vision_reward"
    
    # res_p="res_data/res_vs2_test_sft_17k/VideoReward.json"
    # method_name="video_reward"
    
    # res_p="res_data/res_vs2_test_sft_17k/ImageReward-v1.0.json"
    # method_name="image_reward"
    
    # res_p="res_data/res_vs2_test_sft_17k/LiFT-Critic-13b-lora-v1.5.json"
    # method_name="lift"
    
    # res_p="res_data/res_vs2_test_sft_17k/Q-Align.json"
    # method_name="q_align"
    
    # res_p="res_data/res_vs2_test_sft_17k/DeQA-Score-Mix3.json"
    # method_name="deqa"
    
    res_p="res_data/res_vs2_test_sft_17k/dover.json"
    method_name="dover"
    
    metrics_p=f'metrics_report/report_{method_name}.json'
    from get_acc_corr import get_acc, get_corr
    get_acc(method_name,bench_name,res_p,metrics_p)
    get_corr(method_name,bench_name,res_p,metrics_p)
