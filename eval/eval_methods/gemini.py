import base64
import cv2
from google import genai
from google.genai import types
from typing import List
from utils import extract_video_frames_base64

def gemini_run_one_video(
    user_prompt: str,
    video_path: str,
    chat_config: dict
) -> str | None:
    """
    chat_config example: {
        "model_name":"gemini-2.5-flash"
        "max_tokens":1024
        "temperature":0.7
        "thinking_enabled":True
        "thinking_budget":2048
    }
    """
    # make sure environ 'GOOGLE_API_KEY' is set
    
    try:

        frame_list = extract_video_frames_base64(video_path)

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
        thinking_enabled = chat_config.get("thinking_enabled",False)
        model_name=chat_config.get("model_name", "gemini-2.5-flash")
        
        if thinking_enabled == True:
            config = types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(
                    thinking_budget=chat_config.get("thinking_budget", 2048),
                    include_thoughts=True
                )
            )
        else:
            config = types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(
                    thinking_budget=chat_config.get("thinking_budget", 0)
                ),
                max_output_tokens=chat_config.get("max_tokens", 0.7),
                temperature=chat_config.get("temperature", 0.7)
            )
        
        
        response = client.models.generate_content(
            model=model_name,
            contents=contents,
            config=config
        )
        output=""
        thinking=""
        for part in response.candidates[0].content.parts:
            if not part.text:
                continue
            if part.thought:
                thinking=part.text
                output=part.text
            else:
                output=part.text
                
        if thinking_enabled:
            return "<think>"+thinking+"</think>"+"\n"+output
        else:
            return output
    except Exception as e:
        print(f"[ERROR] Gemini failed: {e}")
        return None
