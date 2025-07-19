import base64
import cv2
import anthropic
from typing import List
from utils import extract_video_frames_base64


def claude_run_one_video(user_prompt: str, video_path: str, chat_config: dict,) -> str | None:
    """
    chat_config example: {
        "model_name":"claude-sonnet-4-20250514"
        "max_tokens":1024
        "temperature":0.7
        "thinking_enabled":False
        "thinking_budget":2048
    }
    """
    # make sure environ 'ANTHROPIC_API_KEY' is set
    try:
        
        client = anthropic.Anthropic()  
        frame_list = extract_video_frames_base64(video_path)

        model_name=chat_config.get("model_name", "claude-sonnet-4-20250514")
        max_tokens=chat_config.get("max_tokens", 1024)
        temperature=chat_config.get("temperature",0.7)
        
        thinking_enabled=chat_config.get("thinking_enabled", False)
        thinking_budget=chat_config.get("thinking_budget", 2048)
        
        messages = [
            {"role": "user", 
            "content": [
                {"type": "text", "text": user_prompt}
                ] + [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}}
                for b64 in frame_list
            ]}
        ]
        
        if thinking_enabled:
            response = client.messages.create(
                model=model_name,
                thinking={
                    "type": "enabled",
                    "budget_tokens": thinking_budget
                },
                max_tokens=max_tokens+thinking_budget,
                temperature=temperature,
                messages=messages
            )
            
        else:
            response = client.messages.create(
                model=model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=messages
            )

        return response.content[0].text

    except Exception as e:
        print(f"[ERROR] GPT run failed: {e}")
        return None