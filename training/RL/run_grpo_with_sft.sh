cd src/r1-v

ID=grpo_27k_2e-6_base_sft_5e-5_960_720
export DEBUG_MODE="true" # Enable Debug if you want to see the rollout of model during RL
export LOG_PATH="./debug_log_$ID.txt"

SFT_Model_Path=videoscore2/vs2_qwen2_5vl_sft_27k_5e-5_2fps_960_720_8192
DATASET_NAME=./Video-R1-data/data_27k_train_rl.json

RUN_NAME="vs2_$ID"
OUTPUT_DIR="./log/$RUN_NAME"

wandb login --relogin $WANDB_API_KEY

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node="4" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12366" \
    src/open_r1/grpo_vs2_sft.py \
    --output_dir ${OUTPUT_DIR} \
    --model_name_or_path ${SFT_Model_Path} \
    --dataset_name ${DATASET_NAME} \
    --deepspeed local_scripts/zero3.json \
    --max_prompt_length 16384 \
    --max_completion_length 1024 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-6 \
    --lr_scheduler_type "cosine" \
    --weight_decay 0.01 \
    --bf16 \
    --gradient_checkpointing true \
    --temporal false \
    --len_control true \
    --attn_implementation flash_attention_2 \
    --max_pixels 589824 \
    --num_train_epochs 1 \
    --run_name ${RUN_NAME} \
    --report_to wandb \
    --log_completions true \
    --logging_steps 2 \
    --save_steps 100 \
    --beta 0.04 \
    --max_grad_norm 5 \
    --save_only_model false \
    --num_generations 8  # number of outputs G in grpo, reduce it would lead to faster training and smaller memory cost but higher variance  
    