import pandas as pd
import json
from datasets import load_dataset
import random
from benchmark import load_benchmark

def judge_equal_for_diverse(method,score1,score2,with_ties):
    # # trival implt
    if with_ties==False:
        if score1 > score2:
            return 1
        elif score1 < score2:
            return -1
        else:
            return 0
    
    else: 
        if method in ["aigve_macs","lift","video_phy2_auto_eval","vs1"]:
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
            



def main(bench,kwargs,short_sampling=False):
    score_json = kwargs["score_json"]
    with_ties = kwargs["with_ties"]
    method = kwargs["method"]
    
    print(method)
    print(bench)
    with open(score_json, 'r') as f:
        score_data = json.load(f)  
    if short_sampling:
        score_data=random.sample(score_data,short_sample_num)
    print("total scores size: ",len(score_data)) 
    
    score_dict = {}
    for item in score_data:
        v_name=item["video_name"]
        if "videophy" in score_json:
            item["v_score_model"]=0
        if None in [item["v_score_model"],item["t_score_model"],item["p_score_model"]]:
            continue
        
        score_dict[v_name]=item
    print("effective scores size: ",len(score_dict))
    print("with ties") if with_ties else print("w/o ties")
        
    # ========================= GenAI-Bench =========================
    if bench in ["genai_bench","genai-bench"]:
        benchmark_data = load_dataset("TIGER-Lab/GenAI-Bench","video_generation",split="test")
        print(f"pairs total num:",len(benchmark_data))
        correct = 0
        total = 0
        for item in benchmark_data:
            left_model = item["left_model"]
            left_video = left_model+"_"+item["left_video"].split("/")[-1].split('.mp4')[0]
            right_model = item["right_model"]
            right_video = right_model+"_"+item["right_video"].split("/")[-1].split('.mp4')[0]
            vote = item["vote_type"] 
            if with_ties==False and vote in ["bothbad_vote","tievote"]:
                continue
            
            if left_video not in score_dict:
                # print("missing video:",left_video)
                continue
            
            if right_video not in score_dict:
                # print("missing video:",left_video)
                continue
            
            total += 1 
            
            score1 = (score_dict[left_video]["v_score_model"]+score_dict[left_video]["t_score_model"]+score_dict[left_video]["p_score_model"])
            score2 = (score_dict[right_video]["v_score_model"]+score_dict[right_video]["t_score_model"]+score_dict[right_video]["p_score_model"])
            
            pred_vote = None
            if judge_equal_for_diverse(method,score1,score2,with_ties) == 1:
                pred_vote = "leftvote"
            elif judge_equal_for_diverse(method,score1,score2,with_ties) == -1:
                pred_vote = "rightvote"
            else:
                if vote == "bothbad_vote":
                    pred_vote = "bothbad_vote"
                else:
                    pred_vote = "tievote"

            if pred_vote == vote:
                correct += 1

        if total > 0:
            acc = correct / total
            print(f"method: {method}\nPairwise accuracy: {correct}/{total} = {acc*100:.3f}\n")
        else:
            print("No valid pairs found.\n")
        
    # ========================= VisionRewardDB-Video =========================
    if bench in ["vision_reward_db_video"]:
        benchmark_data = load_dataset("zai-org/VisionRewardDB-Video", "test")["test"]
        print(f"all pairs num: ",len(benchmark_data))
        correct = 0
        total = 0

        for item in benchmark_data:
            video1 = item["video1_path"].split("/")[-1].split('.mp4')[0]
            video2 = item["video2_path"].split("/")[-1].split('.mp4')[0]
            if video1 not in score_dict or video2 not in score_dict:
                continue
            
            ans = item["standard_answer"]  
            if with_ties==False:
                if ans == "tie":
                    continue

            total += 1 
            score1 = (score_dict[video1]['v_score_model']+score_dict[video1]['t_score_model']+score_dict[video1]['p_score_model'])
            score2 = (score_dict[video2]['v_score_model']+score_dict[video2]['t_score_model']+score_dict[video2]['p_score_model'])
            
            pred_ans = None
            if judge_equal_for_diverse(method,score1,score2,with_ties) == 1:
                pred_ans = "video1"
            elif judge_equal_for_diverse(method,score1,score2,with_ties) == -1:
                pred_ans = "video2"
            else:
                pred_ans = "tie"
                
            if pred_ans == ans:
                correct += 1
        print("current all pairs num:", total)
        if total > 0:
            acc = correct / total
            print(f"method: {method}\nPairwise accuracy: {correct}/{total} = {acc*100:.3f}\n")
        else:
            print("No valid pairs found.\n")
    
    # ========================= VideoGen-Reward-Bench =========================
    if bench in ["videogen_reward_bench","videogen-reward-bench"]:
        csv_path = kwargs["src_file"]
        df = pd.read_csv(csv_path)
        print(f"total pairs num: ",len(df))
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
        print(f"VQ Pairwise Accuracy: {vq_correct}/{total_eff_vq} = {vq_correct/total_eff_vq*100:.3f}")
        print(f"TA Pairwise Accuracy: {ta_correct}/{total_eff_ta} = {ta_correct/total_eff_ta*100:.3f}")
        print(f"Overall Pairwise Accuracy: {overall_correct}/{total_eff_overall} = {overall_correct/total_eff_overall*100:.3f}\n")
    
    # ========================= MJ-Bench-Video =========================
    if bench in ["mj_video_bench","mj_bench_video","mj-video-bench","mj-bench-video"]:
        src_json_path = kwargs["src_file"]
        with open(src_json_path, "r") as f:
            src_data = json.load(f)
        print(f"pairs num: ",len(src_data))

        correct_counts={"Overall":0}
        total = 0
        
        def compare(score1, score2):
            if judge_equal_for_diverse(method,score1,score2,with_ties) == 1:
                return "Video 1 better"
            elif judge_equal_for_diverse(method,score1,score2,with_ties) == -1:
                return "Video 2 better"
            else:
                return "Same"

        for item in src_data:
            video1 = item["video_0_path"].split("/")[-1].split(".")[0]
        
            video2 = item["video_1_path"].split("/")[-1].split(".")[0]
            if video1 not in score_dict:
                # print(f"video {video1} missing")
                continue
            
            if video2 not in score_dict:
                # print(f"video {video2} missing")
                continue
            
            if with_ties == False and item["overall_preference"] == "Same":
                continue
            total+=1
            
            scores1 = score_dict[video1]
            scores2 = score_dict[video2]
            avg1 = (scores1["v_score_model"] + scores1["t_score_model"] + scores1["p_score_model"]) / 3
            avg2 = (scores2["v_score_model"] + scores2["t_score_model"] + scores2["p_score_model"]) / 3
            pred_overall = compare(avg1, avg2)
            if pred_overall == item["overall_preference"]:
                correct_counts["Overall"] += 1
            
        print(f"current pairs num: ",total)
        print(f"method: {method}")
        for cat, correct in correct_counts.items():
            acc = correct / total if total > 0 else 0
            print(f"{cat} Pairwise Accuracy: {correct}/{total} = {acc*100:.3f}")
        print("\n")
        
        
        # category2score = {
        #     "Fineness": "v",
        #     "Alignment": "t",
        #     "Coherence & Consistency": "p"
        # }
        # correct_counts = {cat: 0 for cat in category2score}
        # correct_counts["Overall"] = 0
        # total = 0
        
        # def compare(score1, score2):
        #     if judge_equal_for_diverse(method,score1,score2,with_ties) == 1:
        #         return "Video 1 better"
        #     elif judge_equal_for_diverse(method,score1,score2,with_ties) == -1:
        #         return "Video 2 better"
        #     else:
        #         return "Same"

        # for item in src_data:
        #     video1 = item["video_0_path"].split("/")[-1]
        #     video2 = item["video_1_path"].split("/")[-1]
        #     if video1 not in score_dict or video2 not in score_dict:
        #         continue
        #     scores1 = score_dict[video1]
        #     scores2 = score_dict[video2]
        #     if with_ties == False:
        #         if "Same" in list(item["category_preference"].values()) or item["overall_preference"] == "Same":
        #             continue
        #     total+=1
        #     for category, score_key in category2score.items():
        #         pred = compare(scores1[score_key], scores2[score_key])
        #         gt = item["category_preference"].get(category)
        #         if pred == gt:
        #             correct_counts[category] += 1

        #     avg1 = (scores1["v"] + scores1["t"] + scores1["p"]) / 3
        #     avg2 = (scores2["v"] + scores2["t"] + scores2["p"]) / 3
        #     pred_overall = compare(avg1, avg2)
        #     if pred_overall == item["overall_preference"]:
        #         correct_counts["Overall"] += 1
                
        # print(f"current pairs num: ",total)
        # for cat, correct in correct_counts.items():
        #     acc = correct / total if total > 0 else 0
        #     print(f"method: {method}\n{cat} Pairwise Accuracy: {correct}/{total} = {acc*100:.3f}")
        # print("\n")
    
    # ========================= T2VQA-DB =========================
    if bench in ["t2vqa_db","t2vqa-db"]:
        src_json_path = kwargs["src_file"]
        with open(src_json_path, "r") as f:
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
        print(f"T2VQA-DB Pref Pairwise Accuracy: {correct}/{total} = {acc*100:.3f}")
    
    # ========================= TVGE =========================
    if bench in ["tvge"]:
        src_json_path = kwargs["src_file"]
        with open(src_json_path, "r") as f:
            src_data = json.load(f)
        print(f"pairs num: ",len(src_data))
        total=0
        v_correct=0
        
        for item in src_data:
            video1 = item["video1"]
            video2 = item["video2"]
            if video1 not in score_dict:
                print(f"video {video1} missing")
                continue
            if video2 not in score_dict:
                # print(f"video {video2} missing")
                continue

            if with_ties == False and item["video_quality_pref"] == "same":
                continue
            total+=1
            
            scores1 = score_dict[video1]
            scores2 = score_dict[video2]
            v_score_1 = scores1["v_score_model"]
            v_score_2 = scores2["v_score_model"]
            v_pref=None
            if judge_equal_for_diverse(method,v_score_1,v_score_2,with_ties) == 1:
                v_pref="1"
            elif judge_equal_for_diverse(method,v_score_1,v_score_2,with_ties) == -1:
                v_pref="2"
            else:
                v_pref="same"
            if v_pref == item["video_quality_pref"]:
                v_correct += 1
        
        print(f"method: {method}")
        print(f"current pairs num: ",total)
        acc = v_correct / total if total > 0 else 0
        print(f"TVGE V Pref Pairwise Accuracy: {v_correct}/{total} = {acc*100:.3f}\n")
        
        total=0       
        t_correct=0        
        for item in src_data:
            video1 = item["video1"]
            video2 = item["video2"]
            if video1 not in score_dict:
                print(f"video {video1} missing")
                continue
            if video2 not in score_dict:
                # print(f"video {video2} missing")
                continue
            if with_ties == False and item["text_alignment_pref"] == "same":
                continue
            total+=1
            
            scores1 = score_dict[video1]
            scores2 = score_dict[video2]
            t_score_1 = scores1["t_score_model"]
            t_score_2 = scores2["t_score_model"]
            t_pref=None
            if judge_equal_for_diverse(method,t_score_1,t_score_2,with_ties) == 1:
                t_pref="1"
            elif judge_equal_for_diverse(method,t_score_1,t_score_2,with_ties) == -1:
                t_pref="2"
            else:
                t_pref="same"
                
            if t_pref == item["text_alignment_pref"]:
                t_correct += 1
        
        print(f"current pairs num: ",total)
        acc = t_correct / total if total > 0 else 0
        print(f"TVGE T Pref Pairwise Accuracy: {t_correct}/{total} = {acc*100:.3f}\n")
    
if __name__ == "__main__":
    bench_data_dir="bench_data"
    bench_src_file_mapping={
        "videogen_reward_bench":"bench_data/videogen_reward_bench/videogen-rewardbench.csv",
        
        # "t2vqa_db":"bench_data/t2vqa_db/t2vqa_db_pref.json",
        
        # "vision_reward_db_video":"bench_data/vision_reward_db_video/original_data_vision_reward_db_video.json",
        # "genai_bench":None,
        # "tvge":"bench_data/tvge/tvge_pref.json"
        # "mj_bench_video":"bench_data/mj_bench_video/mj_bench_video_raw.json"
    }
    
    for bench,src_file in bench_src_file_mapping.items():
        
        score_json_mapping={
            # "vs2_float":f"res_data/res_{bench}/vs2_qwen2_5vl_sft_27k_5e-5_2fps_960_720_8192_float_infer_2fps_tempe=0.7.json",
            # "vs2_float":f"res_data/res_{bench}/vs2_qwen2_5vl_sft_27k_no_cot_5e-5_2fps_960_720_8192_infer_2fps_flt_weighted_tempe=0.7.json",
            # "vs2":f"res_data/res_{bench}/vs2_qwen2_5vl_sft_27k_5e-5_2fps_960_720_8192_infer_2fps.json",
            
            # "vs2_float":f"res_data/res_{bench}/vs2_grpo_27k_2e-6_base_qwen2_5_vl_300_float_infer_2fps_tempe=0.7.json",
            # "vs2_float":f"res_data/res_{bench}/vs2_grpo_27k_2e-6_base_qwen2_5_vl_300_infer_2fps_flt_weighted_tempe=0.7.json",
            # "vs2":f"res_data/res_{bench}/vs2_grpo_27k_2e-6_base_qwen2_5_vl_300_infer_2fps.json",
            
            # "vs2_float":f"res_data/res_{bench}/vs2_grpo_27k_2e-6_base_sft_5e-5_960_720_300_float_infer_2fps_tempe=0.7.json",
            # "vs2_float":f"res_data/res_{bench}/vs2_grpo_27k_2e-6_base_sft_5e-5_960_720_200_infer_2fps_flt_normed_tempe=0.7.json",
            "vs2_float":f"res_data/res_{bench}/vs2_grpo_27k_2e-6_base_sft_5e-5_960_720_300_infer_2fps_flt_weighted_tempe=0.7.json",
            # "vs2":f"res_data/res_{bench}/vs2_grpo_27k_2e-6_base_sft_5e-5_960_720_300_infer_2fps.json",
            
            # "aigve_macs":f"res_data/res_{bench}/AIGVE-MACS.json",
            # "deqa":f"res_data/res_{bench}/DeQA-Score-Mix3.json",
            # "dover":f"res_data/res_{bench}/dover.json",
            # "image_reward":f"res_data/res_{bench}/ImageReward-v1.0.json",
            # "lift":f"res_data/res_{bench}/LiFT-Critic-13b-lora-v1.5.json",
            # "q_align":f"res_data/res_{bench}/Q-Align.json",
            # "q_insight":f"res_data/res_{bench}/Q-Insight.json",
            # "video_phy2_auto_eval":f"res_data/res_{bench}/videophy_2_auto.json",
            # "unified_reward":f"res_data/res_{bench}/UnifiedReward-7b.json",
            # "vision_reward":f"res_data/res_{bench}/VisionReward-Video.json",
            # "video_reward":f"res_data/res_{bench}/VideoReward.json",
            # "vs1":f"res_data/res_{bench}/VideoScore-v1.1.json",
            # "vs2":f"res_data/res_{bench}/vs2_qwen2_5vl_sft_17k_2e-4_2fps_512_512_8192_infer_2fps.json",
            # "vs2":f"res_data/res_{bench}/vs2_qwen2_5vl_sft_17k_5e-5_2fps_960_720_8192_infer_2fps.json",
            # "vs2":f"res_data/res_{bench}/vs2_qwen2_5vl_grpo_17k_1e-6_base960-720_reward_3_2400_infer_2fps.json",
            # "vs2_float":f"res_data/res_{bench}/vs2_qwen2_5vl_grpo_17k_1e-6_base960-720_reward_3_2400_float_infer_2fps_tempe=0.7.json",
            
            
        }
        short_sampling=0
        short_sample_num=4000
        
        for method,score_json in score_json_mapping.items():
            kwargs={
                "method": method,
                "src_file":src_file,
                "score_json":score_json,
                # "with_ties":False,
                "with_ties":True,
            }
            load_benchmark(bench_data_dir,bench,num="all")
            main(bench,kwargs,short_sampling)