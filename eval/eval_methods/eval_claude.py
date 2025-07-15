from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import cv2
from template import INPUT_TEMPLATE,DIM_NAMES
import ast
import json


def _get_video_fps(url):
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {url}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps

def infer_one_video(model,processor,video_url,t2v_prompt):
    
    # Messages containing a video url and a text query
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_url,
                },
                {
                    "type": "text", 
                    "text": INPUT_TEMPLATE.substitute(t2v_prompt=t2v_prompt)
                },
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        fps=_get_video_fps(video_url),
        padding=True,
        return_tensors="pt",
        **video_kwargs,
    )
    inputs = inputs.to("cuda")

    # Inference
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print(output_text)
    
    
    
def eval_vs2():
    model_name=""
    processor_name="Qwen/Qwen2.5-VL-7B-Instruct"
    bench_name=""
    bench_data=[]
    eval_res_path=""
    metrics_report_path=""
    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype="auto", device_map="auto"
    )
    
    processor = AutoProcessor.from_pretrained(processor_name)
    res_data=[]
    for item in bench_data:
        video_name=item['video_name']
        video_url=item['video_url']
        t2v_prompt=item['prompt']
        v_score=item['v_score']
        t_score=item['t_score']
        p_score=item['p_score']
        res_item={
            "video_name":item["video_name"],
            "video_url":video_url,
            "prompt":t2v_prompt,
        }
        
        res=infer_one_video(model,processor,video_url,t2v_prompt,)
        res = "{" + res.split("{")[-1].split("}")[0].strip() + "}"
        try:
            eval_res = ast.literal_eval(str(res))
            if any(dim_name not in list(eval_res.keys()) for dim_name in DIM_NAMES):
                print(f"CHECK 0: key error for eval res of {video_name}")
                continue
            res_item["v_score_gt"]=v_score
            res_item["v_score_model"]=eval_res["visual quality"]
            res_item["t_score_gt"]=t_score
            res_item["t_score_model"]=eval_res["text-to-video alignment"]
            res_item["p_score_gt"]=p_score
            res_item["p_score_model"]=eval_res["physical/common-sense consistency"]
            res_data.append(res_item)
        except Exception as e:
            print(e)
            continue
    with open(eval_res_path,"w",encoding='utf-8') as f:
        json.dump(res_data,eval_res_path,indent=4,ensure_ascii=False)
    
    from metrics import compute_accuracy,compute_spcc,compute_plcc
    v_gt=[x["v_score_gt"] for x in res_data]
    v_pred=[x["v_score_model"] for x in res_data]
    
    t_gt=[x["t_score_gt"] for x in res_data]
    t_pred=[x["t_score_model"] for x in res_data]
    
    p_gt=[x["p_score_gt"] for x in res_data]
    p_pred=[x["p_score_model"] for x in res_data]
    
    acc_list=[compute_accuracy(v_gt,v_pred),compute_accuracy(t_gt,t_pred),compute_accuracy(p_gt,p_pred)]
    spcc_list=[compute_spcc(v_gt,v_pred),compute_spcc(t_gt,t_pred),compute_spcc(p_gt,p_pred)]
    plcc_list=[compute_plcc(v_gt,v_pred),compute_plcc(t_gt,t_pred),compute_plcc(p_gt,p_pred)]
    
    
if __name__ == "__main__":
    eval_vs2()