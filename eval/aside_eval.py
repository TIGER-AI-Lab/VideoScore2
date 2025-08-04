



if __name__ == "__main__":
    
    # res_p="res_data/res_vs2_test_sft_17k/open-router-claude-sonnet-4.json"
    # res_p="res_data/res_vs2_test_sft_17k/open-router-claude-sonnet-4_infer_4fps.json"
    # res_p="res_data/res_vs2_test_sft_17k/open-router-gemini-2.5-flash.json"
    # res_p="res_data/res_vs2_test_sft_17k/open-router-gemini-2.5-flash_infer_4fps.json"
    # res_p="res_data/res_vs2_test_sft_17k/open-router-gemini-2.5-pro.json"
    # res_p="res_data/res_vs2_test_sft_17k/open-router-gpt-4.1_infer_4fps.json"
    # res_p="res_data/res_vs2_test_sft_17k/open-router-grok-4_infer_4fps.json"
    # res_p="res_data/res_vs2_test_sft_17k/open-router-o4-mini_infer_4fps.json"
    # res_p="res_data/res_vs2_test_sft_17k/open-router-gemma-3-27b-it_infer_4fps.json"
    # res_p="res_data/res_vs2_test_sft_17k/open-router-llama-4-scout_infer_4fps.json"
    # res_p="res_data/res_vs2_test_sft_17k/open-router-llama-4-maverick_infer_4fps.json"
    # res_p="res_data/res_vs2_test_sft_17k/open-router-qwen2.5-vl-32b-instruct_infer_4fps.json"
    # res_p="res_data/res_vs2_test_sft_17k/open-router-qwen2.5-vl-72b-instruct_infer_4fps.json"
    # res_p="res_data/res_vs2_test_sft_17k/open-router-glm-4.1v-9b-thinking_infer_4fps.json"
    # res_p="res_data/res_vs2_test_sft_17k/vs2_qwen2_5vl_sft_17k_2e-4_8fps_16384_infer_8fps.json"
    # res_p="res_data/res_vs2_test_sft_17k/vs2_qwen2_5vl_sft_17k_2e-4_2fps_512_512_8192_infer_8fps.json"
    # res_p="res_data/res_vs2_test_sft_17k/VideoScore.json"
    # res_p="res_data/res_vs2_test_sft_17k/feat_dino_sim.json"
    # res_p="res_data/res_vs2_test_sft_17k/VisionReward-Video.json"
    # res_p="res_data/res_vs2_test_sft_17k/VideoReward.json"
    # res_p="res_data/res_vs2_test_sft_17k/videophy_2_auto.json"
    # res_p="res_data/res_vs2_test_sft_17k/AIGVE-MACS.json"
    # res_p="res_data/res_vs2_test_sft_17k/vs2_qwen2_5vl_grpo_17k_try_1e-6_800_infer_4fps.json"
    res_p="res_data/res_vs2_test_sft_17k/vs2_qwen2_5vl_grpo_17k_1e-6_base960-720_reward_3_1200_infer_2fps.json"
    # res_p="res_data/res_vs2_test_sft_17k/vs2_qwen2_5vl_grpo_17k_1e-6_reward_3_3200_infer_2fps.json"
    # res_p="res_data/res_vs2_test_sft_17k/vs2_qwen2_5vl_sft_17k_5e-5_2fps_960_720_8192_infer_2fps.json"
    
    method_name="vs2"
    bench_name="vs2_test_sft_17k"
    
    # res_p="res_data/res_video_phy/vs2_qwen2_5vl_sft_17k_2e-4_2fps_512_512_8192_infer_4fps.json"
    # method_name="vs2"
    # bench_name="video_phy"
    
    # res_p="res_data/res_video_phy2/vs2_qwen2_5vl_sft_17k_2e-4_2fps_512_512_8192_infer_4fps.json"
    # method_name="vs2"
    # bench_name="video_phy2"
    
    # res_p="res_data/res_mj_bench_video/vs2_qwen2_5vl_sft_17k_2e-4_2fps_960_720_8192_infer_2fps_old.json"
    # method_name="vs2"
    # bench_name="mj_bench_video"

    # res_p="res_data/res_aigve_bench/vs2_qwen2_5vl_sft_17k_2e-4_2fps_512_512_8192_infer_4fps.json"
    # method_name="vs2"
    # bench_name="aigve_bench"

    # res_p="res_data/res_vs2_test_sft_17k/VisionReward-Video.json"
    # method_name="vision_reward"
    # bench_name="vs2_test_sft_17k"
    
    # res_p="res_data/res_vs2_test_sft_17k/VideoReward.json"
    # method_name="video_reward"
    # bench_name="vs2_test_sft_17k"
    
    # res_p="res_data/res_vs2_test_sft_17k/ImageReward-v1.0.json"
    # method_name="image_reward"
    # bench_name="vs2_test_sft_17k"
    
    metrics_p=f'metrics_report/report_{method_name}.json'
    # from get_acc_corr import get_acc, get_corr
    # get_acc(method_name,bench_name,res_p,metrics_p)
    # get_corr(method_name,bench_name,res_p,metrics_p)
    

    # from transformers import AutoModel, AutoTokenizer, AutoProcessor
    # from transformers import AutoProcessor, AutoModelForVision2Seq
    # from qwen_vl_utils import process_vision_info
    # import torch

    
    # # video_path="/data/xuan/data/videoscore2/videos/vchitect2/001000_p.mp4"
    # # video_path="/data/xuan/data/videoscore2/videos/wanx21_1_3b/000000_v.mp4"
    # video_path="/data/xuan/workdir/VideoScore2/other/example_high_res_videos/000000_r.mp4"
    # video_fps=_get_video_fps(video_path)

    # prompt = "\n<video>\n\nYou are an expert for evaluating and thinking about the quality of AI videos from diverse dimensions.\n\nWe would like to evaluate its quality from three dimensions: 'visual quality', 'text-to-video alignment' and 'physical consistency'. Below is the definition of each dimension: \n(1) \nThe dimension 'visual quality' cares about the video's visual and optical propertities, including 'resolution, overall clarity, local blurriness, smoothness, stability of brightness/contrast, distortion/misalignment, abrupt changes, and any other factors the affect the watching experience'. The keywords written by the annotators are also mostly derived from the above factors.\n\n(2) \nThe dimension 't2v_alignment' mainly assesses whether the generated video fully and accurately depicts the elements mentioned in the text prompt, such as characters, actions, animals, etc., as well as background, quantity, color, weather, and so on. So the keywords written by annotators sometimes only indicate the elements that are missing from the video.\n\n(3) \nThe dimension 'physical/common-sense consistency' mainly examines whether there are any violations of common sense, physical laws, or any other aspects in the video that appear strange or unnatural. Most of the keywords provided by annotators point out the specific abnormalities or inconsistencies they observed in the video.\n\n\nHere we provide an AI video generated by text-to-video models and its text prompt: \nBalancing a scale with a single rich person on one side and thousands of workers on the other, showing the weight of their hard work.\n\nBased on the video content and the dimension definitions, please think about and evaluate the video quality and give the quality score. \nThe quality score must be in 1.0 - 5.0, and the thinking process should be of appropriate length and expression.\n\n"
    
    # model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
    # model = AutoModelForVision2Seq.from_pretrained(
    #     model_name,
    # ).to('cuda')
    # processor = AutoProcessor.from_pretrained(model_name)
    
    # messages = [
    #     {
    #         "role": "user",
    #         "content": [
    #             {
    #                 "type": "video",
    #                 "video": video_path,
    #                 "fps":8.0
    #             },
    #             {
    #                 "type": "text", 
    #                 "text": prompt
    #             },
    #         ],
    #     }
    # ]

    # text = processor.apply_chat_template(
    #     messages, tokenize=False, add_generation_prompt=True
    # )

    # image_inputs, video_inputs = process_vision_info(messages)
    
    # video_frame_pixels = []
    # for i, frame in enumerate(video_inputs):
    #     if isinstance(frame, torch.Tensor):
    #         # 如果frame已被处理为Tensor，shape是 (C, H, W)
    #         print("torch tensor")
    #         res = frame.shape
    #     else:
    #         # 原始PIL图像
    #         res = frame.size
    #     print(res)
    #     # pixels = h * w
    #     # video_frame_pixels.append(pixels)
    #     # print(f"Frame {i}: {w}x{h} = {pixels} pixels")

    # # 获取视频中最大帧的像素数
    # # max_pixels = max(video_frame_pixels)
    # # print(f"Max pixels among all frames: {max_pixels}")
    
    # inputs = processor(
    #     text=[text],
    #     images=image_inputs,
    #     videos=video_inputs,
    #     fps=video_fps,
    #     padding=True,
    #     return_tensors="pt",
    # )
    # inputs = inputs.to("cuda")
    
    # with torch.no_grad():
    #     input_ids = inputs["input_ids"]
    #     # print("input_ids:", input_ids)
    #     print("num_tokens:", input_ids.shape[1])
    