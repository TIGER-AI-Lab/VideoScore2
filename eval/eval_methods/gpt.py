import base64
import json
import re
from typing import List
from openai import OpenAI
from eval_methods.utils import extract_video_frames_base64

def gpt_run_one_video(
    x: dict,
    user_prompt: str,
    video_path: str,
    res_path: str,
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

        # 组织输出结构
        video_name = x["video_name"]
        v_score = x["visual_score"]
        t_score = x["t2v_score"]
        p_score = x["phy_score"]

        res_item = {
            "video_name": video_name,
            "video_url": x["video_url"],
            "prompt": x["prompt"],
            "v_score_gt": v_score,
            "t_score_gt": t_score,
            "p_score_gt": p_score,
        }

        if output is None:
            raise ValueError(f"output for {video_name} is None")

        short_res = output[-100:]
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

        # 写入结果文件
        with open(res_path, "r") as f:
            res_data = json.load(f)
        res_data.append(res_item)
        with open(res_path, "w", encoding="utf-8") as f:
            json.dump(res_data, f, indent=4, ensure_ascii=False)

        print("saved one item")
        return output

    except Exception as e:
        print(f"[ERROR] GPT run one video failed: {e}")
        print(f"Skipped {x.get('video_name', 'unknown')}")
        return None
