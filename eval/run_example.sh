# (1) eval MLLM prompting methods
conda activate vs2_eval
python eval_mllm.py \
    --bench "vs2_bench" \
    --model_name "openai/gpt-5-mini"

# (2) eval reward model / scoring model
# ============================== VideoScore2 ==============================
conda activate vs2_eval
CUDA_VISIBLE_DEVICES=0 python eval_reward_model.py \
    --method "vs2_float" \
    --bench "vs2_bench" \
    --bench_data_num "all" \
    --model_name_or_path "TIGER-Lab/VideoScore2" \
    --kwargs '{"infer_fps":2.0,"max_tokens":4096,"temperature":0.7}'


# ============================== VideoScore1 ==============================
conda activate vs1_eval
CUDA_VISIBLE_DEVICES=0 python eval_reward_model.py \
    --method "vs1" \
    --bench "vs2_bench" \
    --bench_data_num "all" \
    --model_name_or_path "TIGER-Lab/VideoScore"


# ============================== VideoReward ==============================
conda activate video_reward
CUDA_VISIBLE_DEVICES=0 python eval_reward_model.py \
    --method "video_reward" \
    --bench "vs2_bench" \
    --bench_data_num "all" \
    --model_name_or_path "./eval_methods/utils_video_reward/checkpoints/VideoReward"


# ============================== VisionReward ==============================
conda activate vision_reward
CUDA_VISIBLE_DEVICES=0 python eval_reward_model.py \
    --method "vision_reward" \
    --bench "vs2_bench" \
    --bench_data_num "all" \
    --model_name_or_path "THUDM/VisionReward-Video"


# ============================== UnifiedReward ==============================
conda activate unified_reward
CUDA_VISIBLE_DEVICES=0 python eval_reward_model.py \
    --method "unified_reward" \
    --bench "vs2_bench" \
    --bench_data_num "all" \
    --model_name_or_path "CodeGoat24/UnifiedReward-7b" \
    --kwargs '{"infer_fps":2.0,"load_fps":0.0,"num_video_frames":8,"max_tokens":4096}'


# ============================== Q-Align ==============================
conda activate q_align
CUDA_VISIBLE_DEVICES=0 python eval_reward_model.py \
    --method "q_align" \
    --bench "vs2_bench" \
    --bench_data_num "all" \
    --kwargs '{"infer_fps":2.0}'


# ============================== AIGVE-MACS ==============================
conda activate vs2_eval
CUDA_VISIBLE_DEVICES=0 python eval_reward_model.py \
    --method "aigve_macs" \
    --bench "vs2_bench" \
    --bench_data_num "all" \
    --model_name_or_path "xiaoliux/AIGVE-MACS" \
    --kwargs '{"infer_fps":2.0,"max_tokens":1500}'


# ============================== Dover ==============================
conda activate dover
CUDA_VISIBLE_DEVICES=0 python eval_reward_model.py \
    --method "dover" \
    --bench "vs2_bench" \
    --bench_data_num "all" 


# ============================== VideoPhy2-Auto-Eval ==============================
conda activate videophy
CUDA_VISIBLE_DEVICES=0 python eval_reward_model.py \
    --method "video_phy2_auto_eval" \
    --bench "vs2_bench" \
    --bench_data_num "all" \
    --model_name_or_path "./eval_methods/utils_video_phy2/checkpoints/videophy_2_auto" 


# ============================== ImageReward ==============================
conda activate image_reward
CUDA_VISIBLE_DEVICES=0 python eval_reward_model.py \
    --method "image_reward" \
    --bench "vs2_bench" \
    --bench_data_num "all" \
    --model_name_or_path "ImageReward-v1.0" \
    --kwargs '{"infer_fps":2.0,"max_frames":16}'


# ============================== Q-Insight ==============================
conda activate q_insight
CUDA_VISIBLE_DEVICES=0 python eval_reward_model.py \
    --method "q_insight" \
    --bench "vs2_bench" \
    --bench_data_num "all" \
    --model_name_or_path "ByteDance/Q-Insight" \
    --kwargs '{"infer_fps":2.0,"max_frames":16}'


# ============================== DeQA-Score ==============================
conda activate deqa
CUDA_VISIBLE_DEVICES=0 python eval_reward_model.py \
    --method "deqa" \
    --bench "vs2_bench" \
    --bench_data_num "all" \
    --model_name_or_path "zhiyuanyou/DeQA-Score-Mix3" \
    --kwargs '{"infer_fps":2.0,"max_frames":16}'