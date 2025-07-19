import base64
import cv2
from typing import List
from openai import OpenAI
from utils import extract_video_frames_base64

def gpt_run_one_video(
    user_prompt: str,
    video_path: str,
    chat_config: dict
) -> str | None:
    """
    chat_config example: {
        "model_name":"o4-mini"s
        "max_tokens":1024
        "temperature":0.7
        "thinking_enabled":True
        "thinking_effort":"medium"
    }
    """
    
    try:
        base64_frames = extract_video_frames_base64(video_path)

        content = [
            {"type": "text", "text": user_prompt}
        ] + [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{b64}"
                }
            }
            for b64 in base64_frames
        ]

        client = OpenAI()
        model_name=chat_config.get("model_name", "o4-mini")
        thinking_enabled = chat_config.get("thinking_enabled",False)
        if thinking_enabled:
            response = client.responses.create(
                model=model_name,
                reasoning={"effort": chat_config.get("thinking_effort", "medium")},
                input=[{"role": "user", "content": content}]
            )
        else:
            response = client.responses.create(
                model=model_name,
                input=[{"role": "user", "content": content}]
            )
        
        return response.output_text

    except Exception as e:
        print(f"[ERROR] GPT run failed: {e}")
        return None
