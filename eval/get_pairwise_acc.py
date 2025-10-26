import pandas as pd
import json
import random
import os
from benchmark import load_benchmark
import argparse



def judge_equal_for_diverse(method,score1,score2,with_ties):
    ## trival implt
    if with_ties==False:
        if score1 > score2:
            return 1
        elif score1 < score2:
            return -1
        else:
            return 0
    
    else: 
        if method in ["aigve_macs","video_phy2_auto_eval","vs1"]:
            if score1-score2>0:
                return 1
            elif score1-score2<0:
                return -1
            else:
                return 0
        
        if method in ["vs2"]:
            if score1-score2>0.5:
                return 1
            elif score1-score2<-0.5:
                return -1
            else:
                return 0
        
        if method in ["vs2_float"]:
            if score1-score2>0.001:
                return 1
            elif score1-score2<-0.001:
                return -1
            else:
                return 0
            
            
        if method == "video_reward":
            if score1-score2>0.25:
                return 1
            elif score1-score2<-0.25:
                return -1
            else:
                return 0
        
        if method == "unified_reward":
            if score1-score2>0.2:
                return 1
            elif score1-score2<-0.2:
                return -1
            else:
                return 0
            
        if method == "vision_reward":
            if score1-score2>0.05:
                return 1
            elif score1-score2<-0.05:
                return -1
            else:
                return 0

        if method == "deqa":
            if score1-score2>0.5:
                return 1
            elif score1-score2<-0.5:
                return -1
            else:
                return 0
            
        if method == "dover":
            if score1-score2>0.05:
                return 1
            elif score1-score2<-0.05:
                return -1
            else:
                return 0
        
        if method == "image_reward":
            if score1-score2>0.2:
                return 1
            elif score1-score2<-0.2:
                return -1
            else:
                return 0
            
        if method == "q_align":
            if score1-score2>0.05:
                return 1
            elif score1-score2<-0.05:
                return -1
            else:
                return 0
            
        if method == "q_insight":
            if score1-score2>0.05:
                return 1
            elif score1-score2<-0.05:
                return -1
            else:
                return 0
            



def main(bench,method,with_ties,bench_src_file,score_json,):
    
    print(bench)
    with open(score_json, 'r') as f:
        score_data = json.load(f)  
    print("num of total scored items: ",len(score_data)) 
    
    score_dict = {}
    for item in score_data:
        v_name=item["video_name"]
        if "videophy" in score_json:
            item["v_score_model"]=0
        if None in [item["v_score_model"],item["t_score_model"],item["p_score_model"]]:
            continue
        
        score_dict[v_name]=item
    print("effective num of scored items: ",len(score_dict))
    print("with ties") if with_ties else print("w/o ties")
    
    # ========================= VideoGen-Reward-Bench =========================
    if bench in ["videogen_reward_bench","videogen-reward-bench"]:
        df = pd.read_csv(bench_src_file)
        print(f"num of total pairs: ",len(df))
        vq_correct = 0
        ta_correct = 0
        overall_correct = 0
        total_eff_vq=0
        total_eff_ta=0
        total_eff_overall=0
        for _, row in df.iterrows():
            video_A = row["path_A"].split("/")[-1].split('.mp4')[0]
            video_B = row["path_B"].split("/")[-1].split('.mp4')[0]
            if video_A not in score_dict or video_B not in score_dict:
                continue
            
            v_A = score_dict[video_A]["v_score_model"]
            v_B = score_dict[video_B]["v_score_model"]
            if None in [v_A,v_B,]:
                continue
            
            gt_vq = row["VQ"]
            if with_ties == False and gt_vq == "same":
                continue
            
            total_eff_vq+=1
            
            v_pref=None
            if judge_equal_for_diverse(method,v_A,v_B,with_ties) == 1:
                v_pref="A"
            elif judge_equal_for_diverse(method,v_A,v_B,with_ties) == -1:
                v_pref="B"
            else:
                v_pref="same"

            if v_pref == gt_vq:
                vq_correct += 1
         
         
         
         
        for _, row in df.iterrows():
            video_A = row["path_A"].split("/")[-1].split('.mp4')[0]
            video_B = row["path_B"].split("/")[-1].split('.mp4')[0]
            if video_A not in score_dict or video_B not in score_dict:
                continue
            
            t_A = score_dict[video_A]["t_score_model"]
            t_B = score_dict[video_B]["t_score_model"]
            if None in [t_A,t_B,]:
                continue
            
            gt_ta = row["TA"]
            if with_ties == False and gt_ta == "same":
                continue
            
            total_eff_ta+=1
            
            t_pref=None
            if judge_equal_for_diverse(method,t_A,t_B,with_ties) == 1:
                t_pref="A"
            elif judge_equal_for_diverse(method,t_A,t_B,with_ties) == -1:
                t_pref="B"
            else:
                t_pref="same"

            if t_pref == gt_ta:
                ta_correct += 1
                
                
        for _, row in df.iterrows():
            video_A = row["path_A"].split("/")[-1].split('.mp4')[0]
            video_B = row["path_B"].split("/")[-1].split('.mp4')[0]
            if video_A not in score_dict or video_B not in score_dict:
                continue
            
            v_A = score_dict[video_A]["v_score_model"]
            v_B = score_dict[video_B]["v_score_model"]
            t_A = score_dict[video_A]["t_score_model"]
            t_B = score_dict[video_B]["t_score_model"]
            p_A = score_dict[video_A]["p_score_model"]
            p_B = score_dict[video_B]["p_score_model"]
            if None in [v_A,t_A,p_A,v_B,t_B,p_B,]:
                continue
            
            gt_all = row["Overall"]
            if with_ties==False and gt_all == "same":
                continue
            
            total_eff_overall+=1

            overall_pref=None
            if judge_equal_for_diverse(method, v_A+t_A+p_A, v_B+t_B+p_B, with_ties) == 1:
                overall_pref="A"
            elif judge_equal_for_diverse(method, v_A+t_A+p_A, v_B+t_B+p_B, with_ties) == -1:
                overall_pref="B"
            else:
                overall_pref="same"

            if overall_pref == gt_all:
                overall_correct += 1
        
           
        print(f"method: {method}")
        res_vq=f"VideoGen-Reward-Bench VQ Pairwise Accuracy: {vq_correct}/{total_eff_vq} = {vq_correct/total_eff_vq*100:.3f}\n"
        res_ta=f"VideoGen-Reward-Bench TA Pairwise Accuracy: {ta_correct}/{total_eff_ta} = {ta_correct/total_eff_ta*100:.3f}\n"
        res_overall=f"VideoGen-Reward-Bench Overall Pairwise Accuracy: {overall_correct}/{total_eff_overall} = {overall_correct/total_eff_overall*100:.3f}\n"
        print(res_vq)
        print(res_ta)
        print(res_overall)
        return {method:res_vq,method:res_ta,method:res_overall}
    
    # ========================= T2VQA-DB =========================
    if bench in ["t2vqa_db","t2vqa-db"]:
        with open(bench_src_file, "r") as f:
            src_data = json.load(f)
        print(f"pairs num: ",len(src_data))
        total=0
        correct=0
        for item in src_data:
            video1 = item["video1"]
            video2 = item["video2"]
            if video1 not in score_dict:
                print(f"video {video1} missing")
                continue
            if video2 not in score_dict:
                # print(f"video {video2} missing")
                continue
            
            if with_ties == False and item["preference"] == "same":
                continue
            total+=1
            
            scores1 = score_dict[video1]
            scores2 = score_dict[video2]
            avg1 = (scores1["v_score_model"] + scores1["t_score_model"] + scores1["p_score_model"]) / 3
            avg2 = (scores2["v_score_model"] + scores2["t_score_model"] + scores2["p_score_model"]) / 3
            predict_pref=None
            if judge_equal_for_diverse(method,avg1,avg2,with_ties) == 1:
                predict_pref="1"
            elif judge_equal_for_diverse(method,avg1,avg2,with_ties) == -1:
                predict_pref="2"
            else:
                predict_pref="same"
            if predict_pref == item["preference"]:
                correct += 1
        
        print(f"current pairs num: ",total)
        print(f"method: {method}")
        acc = correct / total if total > 0 else 0
        res=f"T2VQA-DB Pref Pairwise Accuracy: {correct}/{total} = {acc*100:.3f}"
        print(res)
        return {method:res}
    
    
if __name__ == "__main__":
    BENCH_DATA_DIR="bench_data"
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench",type=str,default="videogen_reward_bench")
    ap.add_argument("--method_or_model",default="vs2_float")
    ap.add_argument("--with_ties",default=False,)
    args = ap.parse_args()
    bench=args.bench
    method_or_model=args.method_or_model
    with_ties=args.with_ties
    
    bench_src_file_mapping={
        "videogen_reward_bench":"bench_data/videogen_reward_bench/videogen-rewardbench.csv",
        "t2vqa_db":"bench_data/t2vqa_db/t2vqa_db_pref.json",
    }
    
    method_score_json_mapping={
        "vs2_int":f"res_data/res_{bench}/VideoScore2_infer_2fps.json",
        "vs2_float":f"res_data/res_{bench}/VideoScore2_infer_2fps_float_normed_tempe=0.7.json",
        "vs2_float_normed":f"res_data/res_{bench}/VideoScore2_infer_2fps_float_normed_tempe=0.7.json",
        "vs2_float_weighted":f"res_data/res_{bench}/VideoScore2_infer_2fps_float_normed_tempe=0.7.json",
        
        "vs1":f"res_data/res_{bench}/VideoScore.json",
        "aigve_macs":f"res_data/res_{bench}/AIGVE-MACS.json",
        "dover":f"res_data/res_{bench}/dover.json",
        "video_phy2_auto_eval":f"res_data/res_{bench}/videophy_2_auto.json",
        "unified_reward":f"res_data/res_{bench}/UnifiedReward-7b.json",
        "video_reward":f"res_data/res_{bench}/VideoReward.json",
        "vision_reward":f"res_data/res_{bench}/VisionReward-Video.json",
        "q_align":f"res_data/res_{bench}/Q-Align.json",
        "deqa":f"res_data/res_{bench}/DeQA-Score-Mix3.json",
        "image_reward":f"res_data/res_{bench}/ImageReward-v1.0.json",
        "q_insight":f"res_data/res_{bench}/Q-Insight.json",
        
        "claude-sonnet-4":f"res_data/res_{bench}/claude-sonnet-4_infer_2fps_3shot.json",
        "gemini-2.5-flash":f"res_data/res_{bench}/gemini-2.5-flash_infer_2fps_3shot.json",
        "gemini-2.5-pro":f"res_data/res_{bench}/gemini-2.5-pro_infer_2fps_3shot.json",
        "gpt-5":f"res_data/res_{bench}/gpt-5_infer_2fps_3shot.json",
        "gpt-5-mini":f"res_data/res_{bench}/gpt-5-mini_infer_2fps_3shot.json",
        "o4-mini":f"res_data/res_{bench}/o4-mini_infer_2fps_3shot.json",
        "grok-4":f"res_data/res_{bench}/grok-4_infer_2fps_3shot.json",
        "gemma-3-27b-it":f"res_data/res_{bench}/gemma-3-27b-it_infer_2fps_3shot.json",
        "qwen2.5-vl-7b-instruct":f"res_data/res_{bench}/qwen2.5-vl-7b-instruct_infer_2fps.json",
        "qwen2.5-vl-32b-instruct":f"res_data/res_{bench}/qwen2.5-vl-32b-instruct_infer_2fps_3shot.json",
        "qwen2.5-vl-72b-instruct":f"res_data/res_{bench}/qwen2.5-vl-72b-instruct_infer_2fps_3shot.json",
        "llama-4-scout":f"res_data/res_{bench}/llama-4-scout_infer_2fps_3shot.json",
        "llama-4-maverick":f"res_data/res_{bench}/llama-4-maverick_infer_2fps_3shot.json",
        "glm-4.1v-9b-thinking":f"res_data/res_{bench}/glm-4.1v-9b-thinking_infer_2fps_3shot.json",
        "glm-4.5v":f"res_data/res_{bench}/glm-4.5v_infer_2fps_3shot.json",
    }
    
    bench_src_file=bench_src_file_mapping[bench] 
    score_json=method_score_json_mapping[method_or_model]
    metrics_p=f'metrics_report/{bench}/{method_or_model}.json'

    load_benchmark(BENCH_DATA_DIR,bench,num="all")
    res_dict=main(bench,method_or_model,with_ties,bench_src_file,score_json,)
    os.makedirs(os.path.dirname(metrics_p),exist_ok=True)
    with open(metrics_p,"w") as f:
        json.dump(res_dict,f,indent=4)