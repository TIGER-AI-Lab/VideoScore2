import pandas as pd
import json


def main(bench,kwargs):
    if bench in ["videogen_reward_bench","videogen-reward-bench"]:
        csv_path = kwargs["src_csv"]
        json_path = kwargs["score_json"]
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

            v_A = int(score_dict[video_A]["v_score_model"])
            v_B = int(score_dict[video_B]["v_score_model"])
            t_A = int(score_dict[video_A]["t_score_model"])
            t_B = int(score_dict[video_B]["t_score_model"])
            p_A = int(score_dict[video_A]["p_score_model"])
            p_B = int(score_dict[video_B]["p_score_model"])
            
            v_pref = "A" if v_A > v_B else "B"
            t_pref = "A" if t_A > t_B else "B"
            overall_pref = "A" if v_A+t_A+p_A>v_B+t_B+p_B else "B"
            
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
        print(f"Overall Pairwise Accuracy: {ta_correct}/{total} = {ta_correct/total:.3f}")

    if bench in ["genai_bench","genai-bench"]:
        from datasets import load_dataset
        benchmark_data = load_dataset("TIGER-Lab/GenAI-Bench", data_dir="video_generation",split="test")
        with open(json_path, 'r') as f:
            score_data = json.load(f)

        score_dict = {
            item["video_name"]: (
                item["v_score_model"] + item["t_score_model"] + item["p_score_model"]
            ) / 3
            for item in score_data
        }
        correct = 0
        total = 0

        for item in benchmark_data:
            left_video = item["left_video"].split("/")[-1].split('.mp4')[0]
            right_video = item["right_video"].split("/")[-1].split('.mp4')[0]
            vote = item["vote_type"]  
            if left_video not in score_dict or right_video not in score_dict:
                continue
            left_score = score_dict[left_video]
            right_score = score_dict[right_video]
            pred_vote = "leftvote" if left_score > right_score else "rightvote"

            if pred_vote == vote:
                correct += 1
            total += 1

        if total > 0:
            acc = correct / total
            print("Result for GenAI-Bench: ")
            print(f"Pairwise accuracy: {correct}/{total} = {acc:.3f}")
        else:
            print("No valid pairs found.")

        
        
if __name__ == "__main__":
    bench="videogen_reward_bench"
    kwargs={
        "src_csv":"bench_data/videogen_reward_bench/videogen-rewardbench.csv",
        "score_json":"res_data/res_videogen_reward_bench/vs2_qwen2_5vl_sft_17k_2e-4_2fps_512_512_8192_infer_2fps.json",
    }
    
    # bench="genai_bench"
    # kwargs={
    #     "score_json":"res_data/res_genai_bench/vs2_qwen2_5vl_sft_17k_2e-4_2fps_512_512_8192_infer_2fps.json",
    # }
    
    main(bench,kwargs)