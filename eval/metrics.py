
ROUND_DIGIT=5

def compute_accuracy(pred, ground_truth):
    assert len(pred) == len(ground_truth), "len(pred) should be the same as len(ground_truth)"
    
    correct = sum(p == gt for p, gt in zip(pred, ground_truth))
    total = len(ground_truth)
    
    return round(correct / total,ROUND_DIGIT)*100 if total > 0 else 0.0


def compute_accuracy_fuzzy(pred, ground_truth):
    assert len(pred) == len(ground_truth), "len(pred) should be the same as len(ground_truth)"
    
    correct = sum(abs(p-gt)<=1 for p, gt in zip(pred, ground_truth))
    total = len(ground_truth)
    
    return round(correct / total,ROUND_DIGIT)*100 if total > 0 else 0.0
    
def compute_spcc(pred, ground_truth):
    from scipy.stats import spearmanr
    assert len(pred) == len(ground_truth), "len(pred) should be the same as len(ground_truth)"
    coefficient, _ = spearmanr(pred, ground_truth)
    return round(coefficient,ROUND_DIGIT)*100


def compute_plcc(pred, ground_truth):
    from scipy.stats import pearsonr
    assert len(pred) == len(ground_truth), "len(pred) should be the same as len(ground_truth)"
    coefficient, _ = pearsonr(pred, ground_truth)
    return round(coefficient,ROUND_DIGIT)*100


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


def get_metric(method_name,res_p,metric_report_p):
    import json
    import re
    
    
    with open(res_p,"r") as f:
        data=json.load(f)

    v_scores_gt=[x['v_score_gt'] for x in data]
    t_scores_gt=[x['t_score_gt'] for x in data]
    p_scores_gt=[x['p_score_gt'] for x in data]
    v_scores_model=[]
    t_scores_model=[]
    p_scores_model=[]
    
    if all(f"{dim}_score_model" in x for dim in ["v","t","p"] for x in data) :
        v_scores_model=[x['v_score_model'] for x in data]
        t_scores_model=[x['t_score_model'] for x in data]
        p_scores_model=[x['p_score_model'] for x in data]
    
    else:
        pattern = r"visual quality:\s*(\d+).*?text-to-video alignment:\s*(\d+).*?physical/common-sense consistency:\s*(\d+)"
        for x in data:
            video_name=x['video_name']
            output=x['output'][0][-150:]
            try:
                match = re.search(pattern, output, re.DOTALL | re.IGNORECASE)
                if match:
                    v_score=max(int(match.group(1)),1)
                    t_score=max(int(match.group(2)),1)
                    p_score=max(int(match.group(3)),1)
                    
                    v_scores_model.append(v_score)
                    t_scores_model.append(t_score)
                    p_scores_model.append(p_score)
            
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
    
    batch_name="sft_model_score" 
    plot(v_scores_model,batch_name,1)
    plot(t_scores_model,batch_name,2)
    plot(p_scores_model,batch_name,3)
        
    metrics_dict={
        "v_acc":compute_accuracy(v_scores_model,v_scores_gt),
        "t_acc":compute_accuracy(t_scores_model,t_scores_gt),
        "p_acc":compute_accuracy(p_scores_model,p_scores_gt),
        
        "v_acc_fuzzy":compute_accuracy_fuzzy(v_scores_model,v_scores_gt),
        "t_acc_fuzzy":compute_accuracy_fuzzy(t_scores_model,t_scores_gt),
        "p_acc_fuzzy":compute_accuracy_fuzzy(p_scores_model,p_scores_gt),
        
        "v_spcc":compute_spcc(v_scores_model,v_scores_gt),
        "t_spcc":compute_spcc(t_scores_model,t_scores_gt),
        "p_spcc":compute_spcc(p_scores_model,p_scores_gt),
        
        "v_plcc":compute_plcc(v_scores_model,v_scores_gt),
        "t_plcc":compute_plcc(t_scores_model,t_scores_gt),
        "p_plcc":compute_plcc(p_scores_model,p_scores_gt),
        }
    print(metrics_dict)
    # with open(metric_report_p,"w") as f:
    #     json.dump({
    #         method_name:metrics_dict
    #     },f,indent=4)
        

