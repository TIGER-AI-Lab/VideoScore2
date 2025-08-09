import torch
from eval_methods.vs2 import eval_VideoScore2
from string import Template
from benchmark import VS2_QUERY_TEMPLATE

# model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
# model_name = "/data/xuan/workdir/Video-R1/src/r1-v/log/Qwen2.5-VL-7B-GRPO/checkpoint-700"
model_name = "videoscore2/vs2_qwen2_5vl_sft_17k_2e-4_2fps_768_768_8192"
# model_name = "videoscore2/vs2_qwen2_5vl_grpo_17k_1e-6_base960-720_reward_3_2400"
vs2 = eval_VideoScore2(model_name)

video_paths=[
    "/data/xuan/workdir/VideoScore2/other/example_videos/000000_r.mp4"
    # "/data/xuan/workdir/VideoScore2/other/example_videos/001000_s.mp4"
    # "/data/xuan/workdir/VideoScore2/other/example_videos/001500_t.mp4"
    # "/data/xuan/workdir/VideoScore2/other/example_videos/002500_g.mp4"
    # "/data/xuan/workdir/VideoScore2/other/example_videos/p100784.mp4"
]

t2v_prompt = "Ten Lego figures are playing in the park made of building blocks, but they all exploded at the end of the video."
user_prompt=VS2_QUERY_TEMPLATE.substitute(t2v_prompt=t2v_prompt)
method_kwargs = {
    "max_tokens": 1024,     
    "infer_fps": 2.0       
}

for video_path in video_paths:
    with torch.no_grad():
        v_score, t_score, p_score, full_text = vs2.evaluate_video(
            user_prompt=user_prompt,
            video_path=video_path,
            kwargs=method_kwargs
        )

    print(f"Visual Quality Score: {v_score}")
    print(f"Text-to-Video Alignment Score: {t_score}")
    print(f"Physical Consistency Score: {p_score}")
    print("\nFull Output Text:")
    print(full_text)
