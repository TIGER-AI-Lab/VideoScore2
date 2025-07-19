import base64
import cv2
from openai import OpenAI
from typing import List
from utils import extract_video_frames_base64
import json
import requests

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
        base64_str_list = extract_video_frames_base64(video_path)

        if chat_config.get("thinking_enabled", False):
            user_prompt = "You should think step-by-step before answering.\n\n" + user_prompt

        headers = {
            "Authorization": f"Bearer {chat_config['api_key']}",
            "Content-Type": "application/json"
        }
        url = f"{OPEN_ROUTER_URL}/chat/completions"

        payload = {
            "extra_headers": {},
            "extra_body": {},
            "model": chat_config.get("model_name", "anthropic/claude-sonnet-4"),
            "messages": [
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
            ],
            "max_tokens": chat_config.get("max_tokens", 1024),
            "temperature": chat_config.get("temperature", 0.7)
        }
        thinking_enabled=chat_config.get("thinking_enabled",False)
        if thinking_enabled == True:
            payload["reasoning"]={
                "exclude": False,
                "max_tokens": chat_config.get("thinking_budget", 2048)
            },

        response = requests.post(url, headers=headers, data=json.dumps(payload))
        result = response.json()

        thinking = str(result['choices'][0]['message'].get('reasoning', ''))
        output = str(result['choices'][0]['message'].get('content', ''))
        
        if thinking_enabled:
            return "<think>"+thinking+"</think>"+"\n"+output
        else:
            return output
        
    except Exception as e:
        print(f"[ERROR] Failed to process video via OpenRouter Api: {e}")
        return None