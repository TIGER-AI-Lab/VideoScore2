import base64
import json
import re
import cv2
from google import genai
from google.genai import types
from typing import List
from eval_methods.utils import extract_video_frames_base64

def gemini_run_one_video(
    x: dict,
    user_prompt: str,
    video_path: str,
    res_path: str,
    chat_config: dict
) -> str | None:
    """
    chat_config example: {
        "model_name": "gemini-2.5-flash",
        "max_tokens": 1024,
        "temperature": 0.7,
        "thinking_enabled": True,
        "thinking_budget": 2048,
        "infer_fps": 2.0
    }
    """
    try:
        infer_fps = chat_config.get("infer_fps", 2.0)
        frame_list = extract_video_frames_base64(video_path, fps=infer_fps)

        contents = [
            types.ContentPart.text(user_prompt)
        ] + [
            types.ContentPart.inline_data(
                mime_type="image/jpeg",
                data=base64.b64decode(frame_b64)
            )
            for frame_b64 in frame_list
        ]

        client = genai.Client()
        model_name = chat_config.get("model_name", "gemini-2.5-flash")
        max_tokens = chat_config.get("max_tokens", 1024)
        temperature = chat_config.get("temperature", 0.7)
        thinking_enabled = chat_config.get("thinking_enabled", False)
        thinking_budget = chat_config.get("thinking_budget", 2048)

        if thinking_enabled:
            config = types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(
                    thinking_budget=thinking_budget,
                    include_thoughts=True
                ),
                max_output_tokens=max_tokens + thinking_budget,
                temperature=temperature
            )
        else:
            config = types.GenerateContentConfig(
                max_output_tokens=max_tokens,
                temperature=temperature
            )

        response = client.models.generate_content(
            model=model_name,
            contents=contents,
            config=config
        )

        output = ""
        thinking = ""
        for part in response.candidates[0].content.parts:
            if part.thought:
                thinking += part.text
            elif part.text:
                output += part.text

        final_output = f"<think>{thinking}</think>\n{output}" if thinking_enabled else output

        # 结构化结果存储
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

        if final_output is None:
            raise ValueError(f"output for {video_name} is None")
        short_res = final_output[-100:]
        print(short_res)
        print(f"{v_score} {t_score} {p_score}")

        # 提取打分
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

        # 保存
        with open(res_path, "r") as f:
            res_data = json.load(f)
        res_data.append(res_item)
        with open(res_path, "w", encoding='utf-8') as f:
            json.dump(res_data, f, indent=4, ensure_ascii=False)

        print("saved one item")
        return output

    except Exception as e:
        print(f"[ERROR] Gemini run one video failed: {e}")
        print(f"Skipped {x.get('video_name', 'unknown')}")
        return None
