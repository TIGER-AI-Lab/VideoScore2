import base64
import cv2
import anthropic
import json
import re
from typing import List
from eval_methods.utils import extract_video_frames_base64
import os

def claude_run_one_video(
    user_prompt: str,
    video_path: str,
    chat_config: dict
) -> str | None:
    """
    chat_config example:
    {
        "model_name": "claude-sonnet-4-20250514",
        "max_tokens": 1024,
        "temperature": 0.7,
        "thinking_enabled": False,
        "thinking_budget": 2048,
        "infer_fps": 2.0
    }
    """
    try:
        if not os.path.exists(video_path):
            raise ValueError(f"not exist: {video_path}")
        client = anthropic.Anthropic()
        infer_fps = chat_config.get("infer_fps", 2.0)
        frame_list = extract_video_frames_base64(video_path, fps=infer_fps)

        if chat_config.get("thinking_enabled", False):
            user_prompt = "You should think step-by-step before answering.\n\n" + user_prompt

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt}
                ] + [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}}
                    for b64 in frame_list
                ]
            }
        ]

        model_name = chat_config.get("model_name", "claude-sonnet-4-20250514")
        max_tokens = chat_config.get("max_tokens", 1024)
        temperature = chat_config.get("temperature", 0.7)
        thinking_enabled = chat_config.get("thinking_enabled", False)
        thinking_budget = chat_config.get("thinking_budget", 2048)

        if thinking_enabled:
            response = client.messages.create(
                model=model_name,
                thinking={"type": "enabled", "budget_tokens": thinking_budget},
                max_tokens=max_tokens + thinking_budget,
                temperature=temperature,
                messages=messages
            )
            thinking = response.thinking.text if hasattr(response, "thinking") else ""
        else:
            response = client.messages.create(
                model=model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=messages
            )
            thinking = ""

        output = response.content[0].text
        res = f"<think>{thinking}</think>\n{output}" if thinking_enabled else output
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
        print(f"[ERROR] GPT run one video failed: {e}")
        return None
