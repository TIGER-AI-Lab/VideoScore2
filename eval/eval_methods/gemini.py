import base64
import json
import re
import cv2
from google import genai
from google.genai import types
from typing import List
from eval_methods.utils import extract_video_frames_base64

def gemini_run_one_video(
    user_prompt: str,
    video_path: str,
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
        return final_output

    except Exception as e:
        print(f"[ERROR] GPT run one video failed: {e}")
        return None
