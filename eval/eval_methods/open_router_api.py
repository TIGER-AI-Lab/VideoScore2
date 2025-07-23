import base64
import cv2
from openai import OpenAI
from typing import List
from eval_methods.utils import extract_video_frames_base64
import json
import re
import requests

OPEN_ROUTER_URL="https://openrouter.ai/api/v1"


def open_router_run_one_video(
    x: dict,
    user_prompt: str,
    video_path: str,
    res_path: str,
    chat_config: dict
) -> str | None:
    """
    chat_config example: {
        "api_key":'',
        "model_name":"anthropic/claude-sonnet-4"
        "max_tokens":1024
        "temperature":0.7
        "thinking_enabled":False
        "thinking_budget":2048
        
    }
    """
    # try:
    infer_fps=chat_config.get("infer_fps",2.0)
    base64_str_list = extract_video_frames_base64(video_path,infer_fps)

    if chat_config.get("thinking_enabled", False):
        user_prompt = "You should think step-by-step before answering.\n\n" + user_prompt

    headers = {
        "Authorization": f"Bearer {chat_config['api_key']}",
        "Content-Type": "application/json"
    }
    url = f"{OPEN_ROUTER_URL}/chat/completions"

    messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_prompt
                    },
                    *[
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{b64}"
                            }
                        }
                        for b64 in base64_str_list
                    ]
                ]
            }
        ]
    
    payload = {
        "model": chat_config.get("model_name", "anthropic/claude-sonnet-4"),
        "messages": messages, 
        "max_tokens": chat_config.get("max_tokens", 1024),
        "temperature": chat_config.get("temperature", 0.7)
    }
    thinking_enabled=chat_config.get("thinking_enabled",False)
    if thinking_enabled == True:
        payload["reasoning"]={
            "exclude": False,
            "max_tokens": chat_config.get("thinking_budget", 2048)
        }
        payload["max_tokens"] = chat_config.get("max_tokens", 1024) + chat_config.get("thinking_budget", 2048)

    response = requests.post(url, headers=headers, data=json.dumps(payload))
    print(response.json())
    
    thinking = str(response.json()['choices'][0]['message'].get('reasoning', ''))
    output = str(response.json()['choices'][0]['message'].get('content', ''))
    
    if thinking_enabled:
        res = "<think>"+thinking+"</think>"+"\n"+output
    else:
        res = output
    
    video_name=x['video_name']
    v_score=x['visual_score']
    t_score=x['t2v_score']
    p_score=x['phy_score']
    res_item={
        "video_name":x["video_name"],
        "video_url":x['video_url'],
        "prompt":x['prompt'],
        "v_score_gt":v_score,
        "t_score_gt":t_score,
        "p_score_gt":p_score,
    }

    if res is None:
        raise ValueError(f"output for {video_name} is None")
    short_res=res[-100:]
    print(short_res)
    print(f"{v_score} {t_score} {p_score}")
    
    pattern = r"visual quality:\s*(\d+).*?text-to-video alignment:\s*(\d+).*?physical/common-sense consistency:\s*(\d+)"
    match = re.search(pattern, short_res, re.DOTALL | re.IGNORECASE)

    if match:
        res_item["v_score_model"] = int(match.group(1))
        res_item["t_score_model"] = int(match.group(2))
        res_item["p_score_model"] = int(match.group(3))
    else:
        res_item["v_score_model"] = None
        res_item["t_score_model"] = None
        res_item["p_score_model"] = None

    
    res_item["output"]=output
    with open(res_path,"r") as f:
        res_data=json.load(f)
    res_data.append(res_item)
    with open(res_path,"w",encoding='utf-8') as f:
        json.dump(res_data,f,indent=4,ensure_ascii=False)
    print("saved one item")

    # except Exception as e:
    #     print(f"[ERROR] Failed to process video via OpenRouter Api: {e}")
    #     print(f"Skipped {video_name}")
    #     return None