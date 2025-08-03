import pandas as pd
import json
from datasets import load_dataset

def main(bench,kwargs):
    # ========================= VideoGen-Reward-Bench =========================
    if bench in ["videogen_reward_bench","videogen-reward-bench"]:
        csv_path = kwargs["src_csv"]
        json_path = kwargs["score_json"]
        with_ties = kwargs["with_ties"]
        df = pd.read_csv(csv_path)

        with open(json_path, 'r') as f:
            res_data = json.load(f)

        score_dict = {item["video_name"]: item for item in res_data}
        vq_correct = 0
        ta_correct = 0
        overall_correct = 0
        total = 0

        for _, row in df.iterrows():
            video_A = row["path_A"].split("/")[-1].split('.mp4')[0]
            video_B = row["path_B"].split("/")[-1].split('.mp4')[0]
            
            gt_vq = row["VQ"]
            gt_ta = row["TA"]
            gt_all = row["Overall"]
            
            if video_A not in score_dict or video_B not in score_dict:
                continue
            if None in [score_dict[video_A]["v_score_model"],
                        score_dict[video_A]["t_score_model"],
                        score_dict[video_A]["p_score_model"],
                        score_dict[video_B]["v_score_model"],
                        score_dict[video_B]["t_score_model"],
                        score_dict[video_B]["p_score_model"],]:
                continue
            
            v_A = int(score_dict[video_A]["v_score_model"])
            v_B = int(score_dict[video_B]["v_score_model"])
            t_A = int(score_dict[video_A]["t_score_model"])
            t_B = int(score_dict[video_B]["t_score_model"])
            p_A = int(score_dict[video_A]["p_score_model"])
            p_B = int(score_dict[video_B]["p_score_model"])
            
            v_pref=None
            if v_A > v_B:
                v_pref="A"
            elif v_A < v_B:
                v_pref="B"
            else:
                v_pref="same"
            
            t_pref=None
            if t_A > t_B:
                t_pref="A"
            elif t_A < t_B:
                t_pref="B"
            else:
                t_pref="same"

            overall_pref=None
            if v_A+t_A+p_A>v_B+t_B+p_B:
                overall_pref="A"
            elif v_A+t_A+p_A<v_B+t_B+p_B:
                overall_pref="B"
            else:
                overall_pref="same"
            
            if with_ties==False:
                if "same" in [gt_vq, gt_ta, gt_all]:
                    continue
            
            if v_pref == gt_vq:
                vq_correct += 1
            if t_pref == gt_ta:
                ta_correct += 1
            if overall_pref == gt_all:
                overall_correct += 1
            
            total += 1
        print("Result for VideoGen-Reward-Bench: ")
        print(f"VQ Pairwise Accuracy: {vq_correct}/{total} = {vq_correct/total:.3f}")
        print(f"TA Pairwise Accuracy: {ta_correct}/{total} = {ta_correct/total:.3f}")
        print(f"Overall Pairwise Accuracy: {overall_correct}/{total} = {overall_correct/total:.3f}")
    
    # ========================= GenAI-Bench =========================
    elif bench in ["genai_bench","genai-bench"]:
        benchmark_data = load_dataset("TIGER-Lab/GenAI-Bench", data_dir="video_generation",split="test")
        json_path = kwargs["score_json"]
        with_ties = kwargs["with_ties"]
        
        with open(json_path, 'r') as f:
            score_data = json.load(f)
            
        score_dict = {}
        for item in score_data:
            if None in [item["v_score_model"],item["t_score_model"],item["p_score_model"]]:
                continue
            score_dict[item["video_name"]] = (item["v_score_model"] + item["t_score_model"] + item["p_score_model"]) / 3
        correct = 0
        total = 0

        for item in benchmark_data:
            left_model = item["left_model"]
            left_video = left_model+"_"+item["left_video"].split("/")[-1].split('.mp4')[0]
            right_model = item["right_model"]
            right_video = right_model+"_"+item["right_video"].split("/")[-1].split('.mp4')[0]
            vote = item["vote_type"]  
            if left_video not in score_dict or right_video not in score_dict:
                continue
            left_score = score_dict[left_video]
            right_score = score_dict[right_video]
            
            pred_vote = None
            if left_score > right_score:
                pred_vote = "leftvote"
            elif left_score < right_score:
                pred_vote = "rightvote"
            else:
                pred_vote = "bothbad_vote"
                
            if with_ties==False:
                if vote == "bothbad_vote":
                    continue
            
            if pred_vote == vote:
                correct += 1
            total += 1

        if total > 0:
            acc = correct / total
            print("Result for GenAI-Bench: ")
            print(f"Pairwise accuracy: {correct}/{total} = {acc:.3f}")
        else:
            print("No valid pairs found.")

    # ========================= VisionRewardDB-Video =========================
    elif bench in ["vision_reward_db_video"]:
        benchmark_data = load_dataset("zai-org/VisionRewardDB-Video", "test")["test"]
        json_path = kwargs["score_json"]
        with_ties = kwargs["with_ties"]
        with open(json_path, 'r') as f:
            score_data = json.load(f)
            
        score_dict = {}
        for item in score_data:
            if None in [item["v_score_model"],item["t_score_model"],item["p_score_model"]]:
                continue
            score_dict[item["video_name"]] = (item["v_score_model"] + item["t_score_model"] + item["p_score_model"]) / 3
        correct = 0
        total = 0
        for item in benchmark_data:
            video1 = item["video1_path"].split("/")[-1].split('.mp4')[0]
            video2 = item["video2_path"].split("/")[-1].split('.mp4')[0]
            ans = item["standard_answer"]  
            if video1 not in score_dict or video2 not in score_dict:
                continue
            score1 = score_dict[video1]
            score2 = score_dict[video2]
            
            pred_ans = None
            if score1 > score2:
                pred_ans = "video1"
            elif score1 < score2:
                pred_ans = "video2"
            else:
                pred_ans = "tie"
                
            if with_ties==False:
                if vote == "tie":
                    continue
            
            if pred_vote == vote:
                correct += 1
            total += 1

        if total > 0:
            acc = correct / total
            print("Result for GenAI-Bench: ")
            print(f"Pairwise accuracy: {correct}/{total} = {acc:.3f}")
        else:
            print("No valid pairs found.")
    
    # ========================= MJ-Bench-Video =========================
    elif bench in ["mj_video_bench","mj_bench_video","mj-video-bench","mj-bench-video"]:
        src_json_path = kwargs["src_json"]
        score_json_path = kwargs["score_json"]
        with_ties = kwargs["with_ties"]
        with open(src_json_path, "r") as f:
            src_data = json.load(f)
        with open(score_json_path, "r") as f:
            score_data = json.load(f)

        score_dict = {
            item["video_name"]: {
                "v": item["v_score_model"],
                "t": item["t_score_model"],
                "p": item["p_score_model"]
            }
            for item in score_data
        }
        category2score = {
            "Fineness": "v",
            "Alignment": "t",
            "Coherence & Consistency": "p"
        }
        correct_counts = {cat: 0 for cat in category2score}
        correct_counts["Overall"] = 0
        total = 0
        
        def compare(score1, score2):
            if abs(score1 - score2) < 1e-4:
                return "Same"
            return "Video 1 better" if score1 > score2 else "Video 2 better"

        for item in src_data:
            video1 = item["video_0_path"].split("/")[-1]
            video2 = item["video_1_path"].split("/")[-1]
            if video1 not in score_dict or video2 not in score_dict:
                continue
            scores1 = score_dict[video1]
            scores2 = score_dict[video2]
            if with_ties == False:
                if "Same" in list(item["category_preference"].values()) or item["overall_preference"] == "Same":
                    continue
            
            total += 1
            for category, score_key in category2score.items():
                pred = compare(scores1[score_key], scores2[score_key])
                gt = item["category_preference"].get(category)
                if pred == gt:
                    correct_counts[category] += 1

            avg1 = (scores1["v"] + scores1["t"] + scores1["p"]) / 3
            avg2 = (scores2["v"] + scores2["t"] + scores2["p"]) / 3
            pred_overall = compare(avg1, avg2)
            if pred_overall == item["overall_preference"]:
                correct_counts["Overall"] += 1

        for cat, correct in correct_counts.items():
            acc = correct / total if total > 0 else 0
            print(f"{cat} Pairwise Accuracy: {correct}/{total} = {acc:.3f}")

        
if __name__ == "__main__":
    bench="videogen_reward_bench"
    kwargs={
        "src_csv":"bench_data/videogen_reward_bench/videogen-rewardbench.csv",
        "score_json":"res_data/res_videogen_reward_bench/vs2_qwen2_5vl_sft_17k_2e-4_2fps_512_512_8192_infer_2fps.json",
        "with_ties":True
    }

    # bench="genai_bench"
    # kwargs={
    #     "score_json":"res_data/res_genai_bench/vs2_qwen2_5vl_sft_17k_2e-4_2fps_512_512_8192_infer_2fps.json",
    #     "with_ties":False
    # }
    
    # bench="vision_reward_db_video"
    # kwargs={
    #     "score_json":"res_data/res_vision_reward_db_video/vs2_qwen2_5vl_sft_17k_2e-4_2fps_512_512_8192_infer_2fps.json",
    #     "with_ties":False
    # }
    
    main(bench,kwargs)