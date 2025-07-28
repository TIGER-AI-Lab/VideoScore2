import base64
import json
import re
from typing import List
from openai import OpenAI
from eval_methods.utils import extract_video_frames_base64
import os

def gpt_run_one_video(
    user_prompt: str,
    video_path: str,
    chat_config: dict
) -> str | None:
    """
    chat_config example: {
        "model_name":"o4-mini",
        "max_tokens":1024,
        "temperature":0.7,
        "thinking_enabled":True,
        "thinking_effort":"medium",
        "infer_fps": 2.0
    }
    """
    try:
        if not os.path.exists(video_path):
            raise ValueError(f"not exist: {video_path}")
        infer_fps = chat_config.get("infer_fps", 2.0)
        base64_frames = extract_video_frames_base64(video_path, fps=infer_fps)

        content = [{"type": "text", "text": user_prompt}] + [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{b64}"
                }
            }
            for b64 in base64_frames
        ]

        client = OpenAI()
        model_name = chat_config.get("model_name", "o4-mini")
        thinking_enabled = chat_config.get("thinking_enabled", False)

        if thinking_enabled:
            response = client.responses.create(
                model=model_name,
                reasoning={
                    "effort": chat_config.get("thinking_effort", "medium")
                },
                input=[{"role": "user", "content": content}]
            )
        else:
            response = client.responses.create(
                model=model_name,
                input=[{"role": "user", "content": content}]
            )

        output = response.output_text
        return output

    except Exception as e:
        print(f"[ERROR] GPT run one video failed: {e}")
        return None
