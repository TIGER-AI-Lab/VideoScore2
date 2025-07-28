import base64
import cv2
from openai import OpenAI
from typing import List
from eval_methods.utils import extract_video_frames_base64
import json
import re
import requests
import os

OPEN_ROUTER_URL="https://openrouter.ai/api/v1"


def open_router_run_one_video(
    user_prompt: str,
    video_path: str,
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
    try:
        if not os.path.exists(video_path):
            raise ValueError(f"not exist: {video_path}")
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

        thinking = str(response.json()['choices'][0]['message'].get('reasoning', ''))
        output = str(response.json()['choices'][0]['message'].get('content', ''))
        
        if thinking_enabled:
            res = "<think>"+thinking+"</think>"+"\n"+output
        else:
            res = output    
        return res
    
    except Exception as e:
        print(f"[ERROR] Model run one video failed: {e}")
        return None
    
    
