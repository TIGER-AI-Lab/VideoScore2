import base64
import cv2
import anthropic
import json
import re
from typing import List
from eval_methods.utils import extract_video_frames_base64

def claude_run_one_video(
    x: dict,
    user_prompt: str,
    video_path: str,
    res_path: str,
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

        video_name = x['video_name']
        v_score = x['visual_score']
        t_score = x['t2v_score']
        p_score = x['phy_score']
        res_item = {
            "video_name": video_name,
            "video_url": x['video_url'],
            "prompt": x['prompt'],
            "v_score_gt": v_score,
            "t_score_gt": t_score,
            "p_score_gt": p_score,
        }

        if res is None:
            raise ValueError(f"output for {video_name} is None")
        short_res = res[-100:]
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

        res_item["output"] = output

        with open(res_path, "r") as f:
            res_data = json.load(f)
        res_data.append(res_item)
        with open(res_path, "w", encoding='utf-8') as f:
            json.dump(res_data, f, indent=4, ensure_ascii=False)

        print("saved one item")
        return output

    except Exception as e:
        print(f"[ERROR] Claude run one video failed: {e}")
        print(f"Skipped {x['video_name']}")
        return None
