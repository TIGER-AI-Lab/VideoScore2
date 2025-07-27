
# Load model directly
import cv2
from benchmark import VS2_QUERY_TEMPLATE
import os
import requests
from tqdm import tqdm


# from transformers import pipeline

# pipe = pipeline("image-to-text", model="DongfuJiang/vs2_qwen2_5vl_sft_17k")


def _download_file(url: str, save_path: str, overwrite: bool = False, timeout: int = 15):
    chunk_size=1<<14
    if os.path.exists(save_path) and not overwrite:
        print(f"[skip] {save_path} already exists")
        return

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        bar = tqdm(total=total, unit="B", unit_scale=True, desc=os.path.basename(save_path))
        with open(save_path, "wb") as f:
            for chunk in r.iter_content(chunk_size):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
        bar.close()
    print(f"[ok] Downloaded → {save_path}")
    

def _get_video_fps(url):
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {url}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print("video fps: ",fps)
    print("total frames: ", total_frames)
    cap.release()
    return fps


def main():
    video_name="000931_e.mp4"
    video_url="https://molar-public.oss-cn-hangzhou.aliyuncs.com/VideoScore/0500_0999/cogvideox_2b/000931_e.mp4"
    t2v_prompt="Elephant and baby elephant sharing a moment of triumph as they successfully navigate a dense jungle, emphasizing the power of perseverance"
    q_template=VS2_QUERY_TEMPLATE
    
    video_local_path=f"bench_temp/vs2_test_sft_17k/{video_name}"
    _download_file(video_url,video_local_path)
    
    
    # 3 3 2
    from transformers import AutoProcessor, AutoModelForVision2Seq
    from qwen_vl_utils import process_vision_info
    processor = AutoProcessor.from_pretrained("DongfuJiang/vs2_qwen2_5vl_sft_17k")
    model = AutoModelForVision2Seq.from_pretrained("DongfuJiang/vs2_qwen2_5vl_sft_17k").to("cuda")

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_local_path,
                    "fps":8.0
                },
                {
                    "type": "text", 
                    "text": q_template.substitute(t2v_prompt=t2v_prompt)
                },
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        fps=_get_video_fps(video_url),
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")
    
    generated_ids = model.generate(**inputs, max_new_tokens=4096)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print(output_text)

def rewrite_eval_res():
    import json
    import re
    res_p='res_data/temp.json'
    new_res_p='res_data/temp_new.json'
    data=json.load(open(res_p,"r"))
    pattern = r"visual quality:\s*(\d+).*?text-to-video alignment:\s*(\d+).*?physical/common-sense consistency:\s*(\d+)"
    for idx,x in enumerate(data):
        video_name=x['video_name']
        output=x['output'][0]
        try:
            match = re.search(pattern, output[-150:], re.DOTALL | re.IGNORECASE)
            if match:
                x["v_score_model"]=int(match.group(1))
                x["t_score_model"]=int(match.group(2))
                x["p_score_model"]=int(match.group(3))
        
            else:
                print(f"{video_name} no matched score")
                x["v_score_model"]=None
                x["t_score_model"]=None
                x["p_score_model"]=None

        except Exception as e:
            print(f'[err] {e}')
            print(f"{video_name} no matched score")
            x["v_score_model"]=None
            x["t_score_model"]=None
            x["p_score_model"]=None
        x.pop("output",None)
        x["output"]=output
        data[idx]=x
        
    with open(new_res_p,"w",encoding='utf-8') as f:
        json.dump(data,f,indent=4,ensure_ascii=False)

if __name__ == "__main__":
    # main()
    
    from metrics import get_metric
    method_name="vs2_sft_17k"
    
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
    res_p="res_data/res_vs2_test_sft_17k/vs2_qwen2_5vl_sft_17k_2e-4_8fps_16384_infer_8fps.json"
    # res_p="res_data/res_vs2_test_sft_17k/vs2_qwen2_5vl_sft_17k_2e-4_2fps_512_512_8192_infer_8fps.json"
    # res_p="res_data/res_vs2_test_sft_17k/VideoScore.json"
    # res_p="res_data/res_vs2_test_sft_17k/feat_dino_sim.json"
    metrics_p=f'metrics_report/report_{method_name}.json'
    
    # get_metric(method_name,res_p,metrics_p)
    
    
    
    
    
    
    # from transformers import AutoModel, AutoTokenizer, AutoProcessor
    # from transformers import AutoProcessor, AutoModelForVision2Seq
    # from qwen_vl_utils import process_vision_info
    # import torch

    
    # video_path="/data/xuan/videoscore2/videos/kling/000000_r.mp4"
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
    #                 "fps":2.0
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
    