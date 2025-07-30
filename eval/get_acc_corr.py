
ROUND_DIGIT=3

def compute_accuracy(pred, ground_truth):
    assert len(pred) == len(ground_truth), "len(pred) should be the same as len(ground_truth)"
    pred = [0 if x is None else x for x in pred]
    ground_truth  = [-1 if gt is None else gt for gt in ground_truth]
    correct = sum(p == gt for p, gt in zip(pred, ground_truth))
    total = len(ground_truth)
    
    return round(correct / total*100,ROUND_DIGIT) if total > 0 else 0.0


def compute_accuracy_relaxed(pred, ground_truth):
    assert len(pred) == len(ground_truth), "len(pred) should be the same as len(ground_truth)"
    pred = [0 if x is None else x for x in pred]
    ground_truth  = [-1 if gt is None else gt for gt in ground_truth]
    correct = sum(abs(p-gt)<=1 for p, gt in zip(pred, ground_truth))
    total = len(ground_truth)
    
    return round(correct / total *100,ROUND_DIGIT) if total > 0 else 0.0
    

def acc_relaxed_whole_item(pred1,pred2,pred3,gt1,gt2,gt3):
    matched=0
    total = len(gt1)
    assert len(pred1) == len(gt1) and len(pred2) == len(gt2) and len(pred3) == len(gt3), "len(pred) should be the same as len(ground_truth)"
    for p1, p2, p3, g1, g2, g3 in zip(pred1, pred2, pred3, gt1, gt2, gt3):
        if p1 is None or p2 is None or p3 is None:
            continue
        if g1 is None or g2 is None or g3 is None:
            continue
        if abs(p1 - g1) + abs(p2 - g2) + abs(p3 - g3) <= 0 and \
            abs(p1 - g1) <= 1 and \
            abs(p2 - g2) <= 1 and \
            abs(p3 - g3) <= 1:
            matched += 1
            
    return round(matched / total*100,ROUND_DIGIT) if total > 0 else 0.0

    
def compute_spcc(pred, ground_truth):
    filtered_pred = []
    filtered_gt = []
    for ai, bi in zip(pred, ground_truth):
        if ai is not None and bi is not None:
            filtered_pred.append(ai)
            filtered_gt.append(bi)
    
    from scipy.stats import spearmanr
    assert len(filtered_pred) == len(filtered_gt), "len(pred) should be the same as len(ground_truth)"
    coefficient, _ = spearmanr(filtered_pred, filtered_gt)
    coefficient=float(coefficient)
    return round(coefficient *100 ,ROUND_DIGIT)


def compute_plcc(pred, ground_truth):
    filtered_pred = []
    filtered_gt = []
    for ai, bi in zip(pred, ground_truth):
        if ai is not None and bi is not None:
            filtered_pred.append(ai)
            filtered_gt.append(bi)
    
    from scipy.stats import pearsonr
    assert len(filtered_pred) == len(filtered_gt), "len(pred) should be the same as len(ground_truth)"
    coefficient, _ = pearsonr(filtered_pred, filtered_gt)
    coefficient=float(coefficient)
    return round(coefficient*100, ROUND_DIGIT)


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
    
    
def get_acc(method_name,bench_name,score_res_path,metric_report_p):
    v_scores_gt, t_scores_gt, p_scores_gt, v_scores_model, t_scores_model, p_scores_model \
        = load_scores(score_res_path)
    

    # To calculate Accuracy, rescale for different reward models / eval methods
    if method_name in ["aigve_macs"]:
        None
    if method_name in ["vision_reward"]:
        # VisionReward has 1 dim (broadcast to 3), raw score is in [-1, 1]. Rescale to [1,2,3,4,5]  
        v_scores_model = [int(round((x+1)*2.5)) for x in v_scores_model]
        t_scores_model = [int(round((x+1)*2.5)) for x in t_scores_model]
        p_scores_model = [int(round((x+1)*2.5)) for x in p_scores_model]
    if method_name in ["video_reward"]:
        # VideoReward has 2 dim (v t), raw score is in [0, 1]. Rescale to [1,2,3,4,5]  
        v_scores_model = [max(min(int(round(x*5)),5),1) for x in v_scores_model]
        t_scores_model = [max(min(int(round(x*5)),5),1) for x in t_scores_model]
        p_scores_model = [-1 for x in p_scores_model]        
        print(v_scores_model)
    if method_name in ["video_phy2"]:
        # VideoPhy2-AutoEval has 2 dim (t p), score in [1,2,3,4,5]  
        v_scores_model = [-1 for x in v_scores_model]
    
    
    
    # To calculate Accuracy, rescale for different benchmarks
    if "vs2" in bench_name:
        None
    elif bench_name in ["aigve_bench","aigve-bench"]:
        # In AIGVE-Bench, phy dim only has score 1,3,5
        # (1,2)->1, (3,4)->3, 5->5
        p_scores_model=[1 if x == 2 else 3 if x == 4 else x for x in p_scores_model]            
    elif bench_name in ["video_phy","video_phy_test_public"]:
        # In Video-Phy-test, sa and pc dim only have score 0,1
        # (1,2,3)->0, (4,5)->1
        t_scores_model = [0 if x in (1, 2, 3) else 1 for x in t_scores_model]
        p_scores_model = [0 if x in (1, 2, 3) else 1 for x in p_scores_model]
    elif bench_name in ["mj_video_bench","mj_bench_video","mj-video-bench","mj-bench-video"]:
        # In Video-Phy-test, v and t dim have score 0,1,2
        # (1,2)->0, (3,4)->1, 5->2
        t_scores_model = [0 if x in (1, 2) else 1 if x in (3, 4) else 2 for x in t_scores_model]
        p_scores_model = [0 if x in (1, 2) else 1 if x in (3, 4) else 2 for x in p_scores_model]

    
    metrics_dict={
        "v_acc":compute_accuracy(v_scores_model,v_scores_gt),
        "t_acc":compute_accuracy(t_scores_model,t_scores_gt),
        "p_acc":compute_accuracy(p_scores_model,p_scores_gt),
        
        "v_acc_relaxed":compute_accuracy_relaxed(v_scores_model,v_scores_gt),
        "t_acc_relaxed":compute_accuracy_relaxed(t_scores_model,t_scores_gt),
        "p_acc_relaxed":compute_accuracy_relaxed(p_scores_model,p_scores_gt),
        
        "acc_whole_item":acc_relaxed_whole_item(v_scores_model,t_scores_model,p_scores_model,v_scores_gt,t_scores_gt,p_scores_gt),
        }
    print(list(metrics_dict.items())[:3])
    print(list(metrics_dict.items())[3:6])
    print(list(metrics_dict.items())[6:])
    
    # batch_name="sft_model_score" 
    # plot(v_scores_model,batch_name,1)
    # plot(t_scores_model,batch_name,2)
    # plot(p_scores_model,batch_name,3)
    
    # with open(metric_report_p,"w") as f:
    #     json.dump({
    #         method_name:metrics_dict
    #     },f,indent=4)
    
    
    
def get_corr(method_name,bench_name,score_res_path,metric_report_p):
    v_scores_gt, t_scores_gt, p_scores_gt, v_scores_model, t_scores_model, p_scores_model \
        = load_scores(score_res_path)
    
    # rescale for different **reward models / eval methods**
    if method_name in ["aigve_macs"]:
        None
    if method_name in ["vision_reward"]:
        # VisionReward has 1 dim (broadcast to 3), raw score is in [-1, 1]. Rescale to [1,2,3,4,5]  
        # use raw VisionReward score to calculate SPCC/PLCC
        None
    if method_name in ["video_reward"]:
        # VideoReward has 2 dim  (v t), raw score is in [0, 1]. Rescale to [1,2,3,4,5]  
        v_scores_model = [max(min(int(round(x*5)),5),1) for x in v_scores_model]
        t_scores_model = [max(min(int(round(x*5)),5),1) for x in t_scores_model]
        p_scores_model = [-1 for x in p_scores_model]    
    if method_name in ["video_phy2"]:
        # VideoPhy2-AutoEval has 2 dim (t p), score in [1,2,3,4,5]  
        v_scores_model = [-1 for x in v_scores_model]
    
    
    
    # rescale for different **benchmarks**
    if "vs2" in bench_name:
        None
    elif bench_name in ["aigve_bench","aigve-bench"]:
        # In AIGVE-Bench, phy dim only has score 1,3,5
        # (1,2)->1, (3,4)->3, 5->5
        p_scores_model=[1 if x == 2 else 3 if x == 4 else x for x in p_scores_model]            
    elif bench_name in ["video_phy","video_phy_test_public"]:
        # In Video-Phy-test, sa and pc dim only have score 0,1
        # (1,2,3)->0, (4,5)->1
        t_scores_model = [0 if x in (1, 2, 3) else 1 for x in t_scores_model]
        p_scores_model = [0 if x in (1, 2, 3) else 1 for x in p_scores_model]
    elif bench_name in ["mj_video_bench","mj_bench_video","mj-video-bench","mj-bench-video"]:
        # In Video-Phy-test, v and t dim have score 0,1,2
        # (1,2)->0, (3,4)->1, 5->2
        t_scores_model = [0 if x in (1, 2) else 1 if x in (3, 4) else 2 for x in t_scores_model]
        p_scores_model = [0 if x in (1, 2) else 1 if x in (3, 4) else 2 for x in p_scores_model]

    
    metrics_dict={        
        "v_spcc":compute_spcc(v_scores_model,v_scores_gt),
        "t_spcc":compute_spcc(t_scores_model,t_scores_gt),
        "p_spcc":compute_spcc(p_scores_model,p_scores_gt),
        
        "v_plcc":compute_plcc(v_scores_model,v_scores_gt),
        "t_plcc":compute_plcc(t_scores_model,t_scores_gt),
        "p_plcc":compute_plcc(p_scores_model,p_scores_gt),
         }
    print(list(metrics_dict.items())[:3])
    print(list(metrics_dict.items())[3:6])
    
    # batch_name="sft_model_score" 
    # plot(v_scores_model,batch_name,1)
    # plot(t_scores_model,batch_name,2)
    # plot(p_scores_model,batch_name,3)
    
    # with open(metric_report_p,"w") as f:
    #     json.dump({
    #         method_name:metrics_dict
    #     },f,indent=4)
        

