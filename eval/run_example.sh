CUDA_VISIBLE_DEVICES=0 python eval_rm.py \
    --bench "video_phy2_test" \
    --method "vs2_float" \
    --model_name_or_path "TIGER-Lab/VideoScore2" \
    --kwargs '{"infer_fps":2.0}'


