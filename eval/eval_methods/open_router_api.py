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
        "thinking_budget":2048,
        "few_shot_config":{}  # optional
        
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
        if chat_config["few_shot_config"]["enabled"] is True and chat_config["few_shot_config"]["few_shot_egs_path"] is not None:
            import random
            num_egs=chat_config["few_shot_config"].get("num_egs",2)
            few_shot_egs = random.sample(json.load(open(chat_config["few_shot_config"]["few_shot_egs_path"], "r", encoding="utf-8")), num_egs)
            few_shot_msg=[]
            few_shot_fps=1
            for eg in few_shot_egs:
                video_path=eg["video_path"]
                few_shot_msg.extend([
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": eg["user_prompt"]
                            },
                            *[
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{b64}"
                                    }
                                }
                                for b64 in extract_video_frames_base64(video_path,few_shot_fps)
                            ]
                        ]
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": eg["response"]
                            }
                        ]
                    }
                ])
            
            messages = few_shot_msg + messages
            
        payload = {
            "model": chat_config.get("model_name"),
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
        print(f"[INFO] Open Router response status code: {response.status_code}")
        if response.status_code != 200:
            raise ValueError(f"Open Router API error: {response.text}")
        
        thinking = str(response.json()['choices'][0]['message'].get('reasoning', ''))
        output = str(response.json()['choices'][0]['message'].get('content', ''))
        
        if thinking_enabled:
            res = "<think>"+thinking+"</think>"+"\n"+output
        else:
            res = output
            
        pattern = r"visual quality:\s*(\d+).*?text-to-video alignment:\s*(\d+).*?physical/common-sense consistency:\s*(\d+)"
        match = re.search(pattern, res, re.DOTALL | re.IGNORECASE)
        if match:
            v_score_model = int(match.group(1))
            t_score_model = int(match.group(2))
            p_score_model = int(match.group(3))
        else:
            v_score_model = t_score_model = p_score_model = None
        return v_score_model, t_score_model, p_score_model, res    
    
    except Exception as e:
        print(f"[ERROR] Model run one video failed: {e}")
        return None
    
    
