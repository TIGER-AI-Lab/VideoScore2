import os
import json
import torch
import argparse
import logging
import time
import numpy as np
from tqdm import tqdm

from eval_methods.vs2 import eval_VideoScore2
from benchmark import VS2_QUERY_TEMPLATE

NUM_VIDEOS_ALL = 700    
NUM_VIDEOS_QUAL = 500        
GROUP_SIZE = 5         
ROUND_DIGIT = 4


def set_logger(t2v_model,log_name):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    date_time = time.strftime("%m-%d_%H-%M-%S", time.localtime())
    log_name = f'./logs_vs2_bestofn/{t2v_model}_{date_time}.log'
    os.makedirs(os.path.dirname(log_name), exist_ok=True)
    file_handler = logging.FileHandler(log_name)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    return logger


def main(t2v_model, eval_model):
    data_root_dir = "best_of_n"

    prompt_file = f"{data_root_dir}/best_of_n_prompts_500.jsonl"
    prompt_item_list = [json.loads(line) for line in open(prompt_file, "r")]
    
    raw_video_dir = os.path.join(data_root_dir, "videos", t2v_model)
    all_qual_videos=[]
    for seed_idx in range(GROUP_SIZE):
        all_qual_videos.extend(os.listdir(f"{data_root_dir}/videos/{t2v_model}_seed{seed_idx}"))
        
    res_dir = "res_vs2_on_five_videos"
    res_file = os.path.join(data_root_dir, res_dir, f"{t2v_model}.json")
    os.makedirs(os.path.dirname(res_file), exist_ok=True)
    
    grouped_video_list = [[] for _ in range(NUM_VIDEOS_ALL)]
    for video_name in sorted(all_qual_videos):
        group_idx = int(video_name.split("_")[1])
        grouped_video_list[group_idx].append(video_name)

    unzero_count = sum(1 for group in grouped_video_list if group)
    print(f"Number of groups: {unzero_count}")
    
    # ============ resume 部分 ============
    logger = set_logger(t2v_model, f"{data_root_dir}/best_of_n_vs2_logs/{t2v_model}.log")
    vs2_model = eval_VideoScore2(eval_model)
    
    finished_groups = set()
    res_list = []
    if os.path.exists(res_file):
        with open(res_file, "r") as f:
            res_list = json.load(f)
        finished_groups = {item["group_idx"] for item in res_list}
        logger.info(f"Resuming from existing results, already finished {len(finished_groups)} groups.")

    method_kwargs = {"max_tokens": 1024, "infer_fps": 2.0}

    for group_idx, video_group in tqdm(enumerate(grouped_video_list), total=NUM_VIDEOS_ALL):
        if video_group == []:
            continue
        if group_idx in finished_groups:
            continue

        if len(video_group) != GROUP_SIZE:
            logger.error(f"Group {group_idx} size mismatch: {len(video_group)} != {GROUP_SIZE}")
            continue

        s_time_each_group = time.time()
        group_res_dict = {
            "group_idx": group_idx,
            "scores": [],
            "best": ""
        }

        scores_mean = []

        for video in sorted(video_group):
            video_path = os.path.join(raw_video_dir, video)
            prompt = next((item['text'] for item in prompt_item_list if item['idx'] == group_idx), "")
            user_prompt = VS2_QUERY_TEMPLATE.substitute(t2v_prompt=prompt)

            with torch.no_grad():
                v_score, t_score, p_score, _ = vs2_model.evaluate_video(
                    user_prompt=user_prompt,
                    video_path=video_path,
                    kwargs=method_kwargs
                )

            mean_score = float(np.mean([v_score, t_score, p_score]))
            scores_mean.append(mean_score)

            group_res_dict["scores"].append({
                "video": video,
                "aspect_scores": {
                    "visual": v_score,
                    "t2v": t_score,
                    "physical": p_score
                },
                "mean": round(mean_score, ROUND_DIGIT)
            })

        index_best = int(np.argmax(scores_mean))
        group_res_dict["best"] = group_res_dict["scores"][index_best]["video"]

        res_list.append(group_res_dict)
        with open(res_file, "w") as f:
            json.dump(res_list, f, indent=4)

        logger.info(f"[Group {group_idx}] means: {scores_mean}, best index: {index_best}")
        logger.info(f"Time: {time.time() - s_time_each_group:.2f}s")

    logger.info(f"Finished all groups. Results saved in {res_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--t2v_model", type=str, required=True,)
    parser.add_argument("--eval_model", type=str, required=True,)
    args = parser.parse_args()

    main(args.t2v_model, args.eval_model)

    
    