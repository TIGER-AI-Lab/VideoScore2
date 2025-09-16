
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


def _load_scores(method_name,bench_name,score_res_path):
    import json
    import re
    with open(score_res_path,"r") as f:
        data=json.load(f)
    
    overall_scores_gt=[0 for x in data]
    if bench_name in ["mj_bench_video"]:
        if all('total_score' in x for x in data):
            overall_scores_gt=[x['total_score'] for x in data]
    
    if bench_name in ["t2vqa_db"]:
        if all('quality_score' in x for x in data):
            overall_scores_gt=[x['quality_score'] for x in data]
    
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
    
    if method_name in ["video_phy2_auto_eval"]:
        v_scores_model=[0 for x in data]
    if method_name in ["video_reward","lift"]:
        p_scores_model=[0 for x in data]
    
    overall_scores_gt, v_scores_gt, t_scores_gt, p_scores_gt, v_scores_model, t_scores_model, p_scores_model = \
        _remove_null_scores(overall_scores_gt, v_scores_gt, t_scores_gt, p_scores_gt, v_scores_model, t_scores_model, p_scores_model)
    
    return overall_scores_gt, v_scores_gt, t_scores_gt, p_scores_gt, v_scores_model, t_scores_model, p_scores_model
    
    
def _remove_null_scores(overall_scores_gt, v_scores_gt, t_scores_gt, p_scores_gt, v_scores_model, t_scores_model, p_scores_model):
    new_overall_gt = [] 
    new_v_gt = [] 
    new_t_gt = []
    new_p_gt = []
    new_v_model = []
    new_t_model = []
    new_p_model = []
    removed_num=0
    for overall_gt, v_gt, t_gt, p_gt, v_m, t_m, p_m in zip(overall_scores_gt, v_scores_gt, t_scores_gt, p_scores_gt, v_scores_model, t_scores_model, p_scores_model):
        if None not in (v_m, t_m, p_m):
            new_v_gt.append(v_gt)
            new_t_gt.append(t_gt)
            new_p_gt.append(p_gt)
            new_v_model.append(v_m)
            new_t_model.append(t_m)
            new_p_model.append(p_m)
            new_overall_gt.append(overall_gt)
        else:
            removed_num+=1
    if removed_num>0:
        print(f"[warn] removed {removed_num} null items when loading scores.")
    return new_overall_gt, new_v_gt, new_t_gt, new_p_gt, new_v_model, new_t_model, new_p_model

def _compute_accuracy(pred, ground_truth):
    if len(pred) == 0 or all(x is None for x in pred):
        print(f"[warn] empty or null pred, return 0.0")
        return 0.0
    if len(ground_truth) == 0:
        print(f"[warn] empty ground truth, return 0.0")
        return 0.0
        
    assert len(pred) == len(ground_truth), "len(pred) should be the same as len(ground_truth)"
    try:
        pred = [0 if x is None else x for x in pred]
        ground_truth  = [-1 if gt is None else gt for gt in ground_truth]
        correct = sum(p == gt for p, gt in zip(pred, ground_truth))
        total = len(ground_truth)
        return round(correct / total*100,ROUND_DIGIT) if total > 0 else 0.0
    
    except Exception as e:
        print(e)
        return 0.0


def _compute_accuracy_relaxed(pred, ground_truth):
    if len(pred) == 0 or any(x is None for x in pred):
        print(f"[warn] empty or null pred, return 0.0")
        return 0.0
    if len(ground_truth) == 0:
        print(f"[warn] empty ground truth, return 0.0")
        return 0.0
    
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

def _acc_whole_item(pred1,pred2,pred3,gt1,gt2,gt3):
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
        
    
def _compute_spcc(pred, ground_truth):
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


def _compute_plcc(pred, ground_truth):
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
    
    overall_scores_gt, v_scores_gt, t_scores_gt, p_scores_gt, v_scores_model, t_scores_model, p_scores_model \
        = _load_scores(method_name,bench_name,score_res_path)
    overall_scores_model=[None for _ in overall_scores_gt]
    
    # To calculate Accuracy, rescale for different reward models / eval methods
    if method_name in ["vs2"]:
        None
    
    if method_name in ["vs2_float"]:
        v_scores_model = [int(min(5, max(1, round(x)))) for x in v_scores_model]
        t_scores_model = [int(min(5, max(1, round(x)))) for x in t_scores_model]
        p_scores_model = [int(min(5, max(1, round(x)))) for x in p_scores_model]
    
    if method_name in ["unified_reward"]:
        # UnifiedReward (the version for video generation point score) has 1 dim (broadcast to 3).
        # Raw score (float) in [1,4]. Rescale to [1,2,3,4,5].      
        v_scores_model = t_scores_model = p_scores_model =[
            int(min(5, max(1, round(1.25*x)))) for x in v_scores_model
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
        p_scores_model = [None for x in p_scores_model]        
    
    if method_name in ["image_reward"]:
        # ImageReward has 1 dim (final score), broadcast to 3 dim. 
        # Raw score (float). Normalized to have mean=1 and std=1. Assume Gaussian Dist. Rescale to [1,2,3,4,5]  
        from scipy.stats import norm 
        v_scores_model = t_scores_model = p_scores_model = [
            1 if z < norm.ppf(0.2) else 2 if z < norm.ppf(0.4) else 3 if z < norm.ppf(0.6) else 4 if z < norm.ppf(0.8) else 5
            for z in v_scores_model
        ]
    
    if method_name in ["aigve_macs"]:
        v_scores_model = [min(5, max(1, round(x))) for x in v_scores_model]    
        t_scores_model = [min(5, max(1, round(x))) for x in t_scores_model]
        p_scores_model = [min(5, max(1, round(x))) for x in p_scores_model]
        
    if method_name in ["lift"]:
        # LiFT has 3 dim {fiedlity, semantic, motion}. 
        # Raw score (int) in [1,2,3]. 
        if bench_name in ["aigve_bench","mj_bench_video","video_phy2"]:
            # the above benchmarks have <=3 score levels, which LiFT score can be adapted.
            # For other benchmarks, skip LiFT for acc calculation
            v_scores_model=[1 if x==1 else 3 if x==2 else 5 for x in v_scores_model]
            t_scores_model=[1 if x==1 else 3 if x==2 else 5 for x in t_scores_model]
            p_scores_model=[None for x in p_scores_model]
        else:
            print("[skip] skipping acc calculation for method LiFT. Exited")      
            return None
        
    if method_name in ["video_phy2_auto_eval"]:
        # VideoPhy2-AutoEval has 2 dim (t p). Raw score (int) in [1,2,3,4,5]. Some scores are 0.
        v_scores_model = [None for x in v_scores_model]    
        t_scores_model = [min(5, max(1, round(x))) for x in t_scores_model]
        p_scores_model = [min(5, max(1, round(x))) for x in p_scores_model]
        
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

    
    
    # To calculate Accuracy, rescale for some different benchmarks   
    if "vs2" in bench_name:
        None
         
    if bench_name in ["aigve_bench",]:
        # In AIGVE-Bench, phy dim only has score 1,3,5
        # (1,2)->1, (3,4)->3, 5->5
        p_scores_model=[1 if x in [1,2] else 3 if x in [3,4] else 5 if x in [5] else None for x in p_scores_model]            
    
    
    if bench_name in ["video_phy_test",]:
        # In Video-Phy-test, sa and pc dim only have score 0,1
        # (1,2,3)->0, (4,5)->1
        t_scores_model = [0 if x in [1, 2, 3] else 1 if x in [4, 5] else None for x in t_scores_model]
        p_scores_model = [0 if x in [1, 2, 3] else 1 if x in [4, 5] else None for x in p_scores_model]
        
        
    if bench_name in ["mj_bench_video",]:
        # In MJ-Bench-Video, v t p dim all have score 0,1,2
        # (1,2)->0, (3,4)->1, 5->2
        
        import json
        with open(score_res_path,"r") as f:
            tmp_data=json.load(f)
        
        if method_name in ["video_phy2_auto_eval"]:
            overall_scores_model=[int((t+p)/2) for _,t,p in zip(v_scores_model,t_scores_model,p_scores_model)]
        elif method_name in ["lift","video_reward"]:
            overall_scores_model=[int((v+t)/2) for v,t,_ in zip(v_scores_model,t_scores_model,p_scores_model)]
        else:
            overall_scores_model=[int((v+t+p)/3) for v,t,p in zip(v_scores_model,t_scores_model,p_scores_model)]
        
        
        v_scores_model = [0 if x in [1,2] else 1 if x in [3, 4] else 2 if x in [5] else None for x in v_scores_model]
        t_scores_model = [0 if x in [1] else 1 if x in [2, 3] else 2 if x in [4, 5] else None for x in t_scores_model]
        p_scores_model = [0 if x in [1] else 1 if x in [2, 3] else 2 if x in [4, 5] else None for x in p_scores_model]
        overall_scores_model=[0 if x in [1] else 1 if x in [2, 3] else 2 for x in overall_scores_model]


    if bench_name in ["tvge","t2v_gen_eval"]:
        # In TVGE (or T2V Gen Eval), 2 dim v and t. Scores in Benchmark are float in [1.0, 5.0].
        # Skip acc calculation for this benchmark.
        print("[skip] skipping acc calculation for benchmark TVGE.")    
        return None
    
    
    if bench_name in ["t2vqa_db"]:
        # In T2VQA-DB, only one final score. Scores in Benchmark are float in [0,100].
        # Skip acc calculation for this benchmark.
        print("[skip] skipping acc calculation for benchmark T2VQA-DB.")    
        return None
    
    # import random
    # random.seed(42)
    # v_scores_model=[random.choice([1,2,3,4,5]) for _ in v_scores_gt]
    # t_scores_model=[random.choice([1,2,3,4,5]) for _ in t_scores_gt]
    # p_scores_model=[random.choice([1,2,3,4,5]) for _ in p_scores_gt]
    
    metrics_dict={
        "v_acc":_compute_accuracy(v_scores_model,v_scores_gt),
        "t_acc":_compute_accuracy(t_scores_model,t_scores_gt),
        "p_acc":_compute_accuracy(p_scores_model,p_scores_gt),
        
        "v_acc_relaxed":_compute_accuracy_relaxed(v_scores_model,v_scores_gt),
        "t_acc_relaxed":_compute_accuracy_relaxed(t_scores_model,t_scores_gt),
        "p_acc_relaxed":_compute_accuracy_relaxed(p_scores_model,p_scores_gt),
        
        "overall_acc":_compute_accuracy(overall_scores_model,overall_scores_gt),
        
        "acc_whole_item":_acc_whole_item(v_scores_model,t_scores_model,p_scores_model,v_scores_gt,t_scores_gt,p_scores_gt),
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
    
    overall_scores_gt, v_scores_gt, t_scores_gt, p_scores_gt, v_scores_model, t_scores_model, p_scores_model \
        = _load_scores(method_name,bench_name,score_res_path)
    
    overall_scores_model=[]
    
    # rescale for different **reward models / eval methods**
    if method_name in ["vs2","vs2_float","vs1",
                       "vision_reward","unified_reward","image_reward",
                       "aigve_macs","dover",
                       "lift",
                       "q_insight","q_align","deqa"]:
        # use raw score to calculate SPCC/PLCC
        None 
        
    if method_name in ["video_reward"]:
        # VideoReward has 2 dim (v t), ignore dim3
        # use raw VideoReward score to calculate SPCC/PLCC
        p_scores_model = [None for x in p_scores_model]
        
    if method_name in ["video_phy2_auto_eval"]:
        # VideoPhy2-AutoEval has 2 dim (t p), ignore dim1
        # use raw VideoPhy2-AutoEval score to calculate SPCC/PLCC
        v_scores_model = [None for x in v_scores_model]




    # rescale for different **benchmarks**
    if "vs2" in bench_name:
        None
    
    if bench_name in ["aigve_bench",]:
        None
    
    if bench_name in ["video_phy_test","video_phy2_test"]:
        v_scores_model = [None for x in v_scores_model]
        v_scores_gt = [None for x in v_scores_gt]
        
    
    if bench_name in ["tvge",]:
        p_scores_model = [None for x in p_scores_model]
        p_scores_gt = [None for x in p_scores_gt]
        
    
    if bench_name in ["mj_bench_video", "t2vqa_db"]:        
        if method_name in ["video_phy2_auto_eval"]:
            overall_scores_model=[t+p for _,t,p in zip(v_scores_model,t_scores_model,p_scores_model)]
        elif method_name in ["lift","video_reward"]:
            overall_scores_model=[v+t for v,t,_ in zip(v_scores_model,t_scores_model,p_scores_model)]
        else:
            overall_scores_model=[v+t+p for v,t,p in zip(v_scores_model,t_scores_model,p_scores_model)]
        
    import random
    random.seed(44)
    v_scores_model=[random.choice([1,2,3,4,5]) for _ in v_scores_gt]
    t_scores_model=[random.choice([1,2,3,4,5]) for _ in t_scores_gt]
    p_scores_model=[random.choice([1,2,3,4,5]) for _ in p_scores_gt]
    
    metrics_dict={        
        "v_spcc":_compute_spcc(v_scores_model,v_scores_gt),
        "t_spcc":_compute_spcc(t_scores_model,t_scores_gt),
        "p_spcc":_compute_spcc(p_scores_model,p_scores_gt),
        "overall_spcc":_compute_spcc(overall_scores_model,overall_scores_gt),
            
        "v_plcc":_compute_plcc(v_scores_model,v_scores_gt),
        "t_plcc":_compute_plcc(t_scores_model,t_scores_gt),
        "p_plcc":_compute_plcc(p_scores_model,p_scores_gt),
        "overall_plcc":_compute_plcc(overall_scores_model,overall_scores_gt),
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
    bench="vs2_test_sft_27k"
    # bench="mj_bench_video"
    # bench="aigve_bench"
    # bench="video_phy2_test"
    # bench="tvge"
    # bench="t2vqa_db"
    
    res_path_mapping={
        # "vs2":f"res_data/res_{bench}/vs2_qwen2_5vl_sft_17k_2e-4_2fps_512_512_8192_infer_2fps.json",
        # "vs2":f"res_data/res_{bench}/vs2_qwen2_5vl_sft_17k_2e-4_2fps_960_720_8192_infer_2fps.json",
        # "vs2":f"res_data/res_{bench}/vs2_qwen2_5vl_grpo_17k_1e-6_base960-720_reward_3_2400_infer_2fps.json",
        # "vs2":f"res_data/res_{bench}/vs2_qwen2_5vl_grpo_17k_5e-7_base960_720_reward4_temporal_2400_infer_2fps.json",
        # "vs2":f"res_data/res_{bench}/vs2_qwen2_5vl_grpo_17k_1e-6_base960_720_reward4_temporal_2400_infer_2fps.json",
        # "vs2_float":f"res_data/res_{bench}/vs2_qwen2_5vl_grpo_17k_1e-6_base960-720_reward_3_2400_float_infer_2fps_tempe=0.9.json",
        # "aigve_macs":f"res_data/res_{bench}/AIGVE-MACS.json",
        # "deqa":f"res_data/res_{bench}/DeQA-Score-Mix3.json",
        # "dover":f"res_data/res_{bench}/dover.json",
        # "image_reward":f"res_data/res_{bench}/ImageReward-v1.0.json",
        # "lift":f"res_data/res_{bench}/LiFT-Critic-13b-lora-v1.5.json",
        # "video_phy2_auto_eval":f"res_data/res_{bench}/videophy_2_auto.json",
        # "unified_reward":f"res_data/res_{bench}/UnifiedReward-7b.json",
        "video_reward":f"res_data/res_{bench}/VideoReward.json",
        # "vision_reward":f"res_data/res_{bench}/VisionReward-Video.json",
        # "q_align":f"res_data/res_{bench}/Q-Align.json",
        # "q_insight":f"res_data/res_{bench}/Q-Insight.json",
        
        # "claude-sonnet-4":f"res_data/res_{bench}/open-router-claude-sonnet-4_infer_2fps.json",
        # "gemini-2.5-flash":f"res_data/res_{bench}/open-router-gemini-2.5-flash_infer_2fps.json",
        # "gemini-2.5-pro":f"res_data/res_{bench}/open-router-gemini-2.5-pro_infer_2fps.json",
        # "gpt-5":f"res_data/res_{bench}/open-router-gpt-5_infer_2fps.json",
        # "gpt-5-mini":f"res_data/res_{bench}/open-router-gpt-5-mini_infer_2fps.json",
        # "o4-mini":f"res_data/res_{bench}/open-router-o4-mini_infer_2fps.json",
        # "grok-4":f"res_data/res_{bench}/open-router-grok-4_infer_2fps.json",
        # "gemma-3-27b-it":f"res_data/res_{bench}/open-router-gemma-3-27b-it_infer_2fps.json",
        # "qwen2.5-vl-72b-instruct":f"res_data/res_{bench}/open-router-qwen2.5-vl-72b-instruct_infer_2fps.json",
        # "llama-4-scout":f"res_data/res_{bench}/open-router-llama-4-scout_infer_2fps.json",
        # "glm-4.1v-9b-thinking":f"res_data/res_{bench}/open-router-glm-4.1v-9b-thinking_infer_2fps.json",
    }
    
    for method_name, res_p in res_path_mapping.items():
        metrics_p=f'metrics_report/report_{method_name}.json'
        from get_acc_corr import get_acc, get_corr
        print(f"method_name: {method_name}")
        get_acc(method_name,bench,res_p,metrics_p)
        get_corr(method_name,bench,res_p,metrics_p)
        print("\n")
        
    