
# Load model directly
from transformers import AutoProcessor, AutoModelForVision2Seq
from qwen_vl_utils import process_vision_info
import cv2
from template import INPUT_TEMPLATE
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
    cap.release()
    return fps


def main():
    video_name="000931_e.mp4"
    video_url="https://molar-public.oss-cn-hangzhou.aliyuncs.com/VideoScore/0500_0999/cogvideox_2b/000931_e.mp4"
    t2v_prompt="Elephant and baby elephant sharing a moment of triumph as they successfully navigate a dense jungle, emphasizing the power of perseverance"
    q_template=INPUT_TEMPLATE
    
    video_local_path=f"bench_temp/vs2_test_sft_17k/{video_name}"
    _download_file(video_url,video_local_path)
    
    # 3 3 2
    
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
    res_p='res_data/temp.json'
    metrics_p=f'metrics_report/report_{method_name}.json'
    
    get_metric(method_name,res_p,metrics_p)
    
    