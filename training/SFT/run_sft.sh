echo "Start training..."

wandb login --relogin $WANDB_API_KEY
llamafactory-cli train examples/train_full/vs2_qwen2_5vl_sft_27k_5e-5_2fps_960_720_8192.yaml \
    hf_hub_token=$HF_TOKEN \
    dataset=data_27k_train_sft