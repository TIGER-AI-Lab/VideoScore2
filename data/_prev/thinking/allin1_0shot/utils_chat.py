import base64
import os
import requests
from PIL import Image
import io
import warnings
import time


def _thinking_cmt_claude(model_access,raw_comments,prompt,frame_list,template,def1,def2,def3)->dict:
    import anthropic
    import httpx
    s_t=time.time()
    os.environ["ANTHROPIC_API_KEY"]=model_access["api_key"]
    model_name=model_access["model_name"]
    def claude_img_input(path_url_pil)->str:
        if isinstance(path_url_pil,str) and "http" in path_url_pil:
            base64_str = base64.standard_b64encode(httpx.get(path_url_pil).content).decode("utf-8")
            return base64_str
        elif isinstance(path_url_pil,str) and not "http" in path_url_pil:
            with open(path_url_pil, "rb") as img_file:
                base64_str=base64.standard_b64encode(img_file.read()).decode("utf-8")
            return base64_str
        elif isinstance(path_url_pil,Image.Image):
            buffered = io.BytesIO()
            img = path_url_pil.convert("RGB")  
            img.save(buffered, format="JPEG")  
            base64_str = base64.standard_b64encode(buffered.getvalue()).decode("utf-8")
            return base64_str
        else:
            warnings.warn("image input is not url, path or PIL Image!")
            return ""
    
    client = anthropic.Anthropic()
    
    # only text
    if len(frame_list)==0 or frame_list is None:
        try:
            completion = client.messages.create(
                model=model_name,
                max_tokens=8000,
                thinking={
                    "type": "enabled",
                    "budget_tokens": 5000
                },
                system="You are an expert for evaluating and commenting on the quality of AI videos.",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": template.substitute(visual_def=def1,t2v_def=def2,phy_def=def3,
                                                            comment_visual=raw_comments[0],
                                                            comment_t2v=raw_comments[1],
                                                            comment_phy=raw_comments[2],
                                                            prompt=prompt)
                            }
                        ]
                    }
                ]
            )
        except Exception as e:
            print(e)
            
        try:
            thinking = str(completion.content[0].thinking)
            output = str(completion.content[1].text)
            return {
                "thinking": thinking,
                "output": output
            }

        except Exception as e:
            return {
                "thinking": None,
                "output": None
            }
        
    # text attached with images
    else:
        base64_str_list = [claude_img_input(path) for path in frame_list]
        try:
            completion = client.messages.create(
                model=model_name,
                max_tokens=8000,
                thinking={
                    "type": "enabled",
                    "budget_tokens": 5000
                },
                system="You are an expert for evaluating and thinking about the quality of AI videos from diverse dimensions.",
                messages=[
                    {
                        "role":"user",
                        "content": [
                            {
                                "type": "text", 
                                "text": template.substitute(visual_def=def1,t2v_def=def2,phy_def=def3,
                                                            comment_visual=raw_comments[0],
                                                            comment_t2v=raw_comments[1],
                                                            comment_phy=raw_comments[2],
                                                            prompt=prompt)\
                                +"\n\nHere is some frames of the video for your reference, If there is any discrepancy between the manual annotation and the video frames, please refer to the video frames as the standard.",
                            },
                            *[
                                {
                                    "type": "image",
                                    "source":{
                                        "type":"base64",  
                                        "media_type":"image/jpeg",
                                        "data":base64_str
                                    }
                                }  
                                for _,base64_str in zip(frame_list, base64_str_list)
                            ]
                        ],
                    },
                ]
            )
        except Exception as e:
            print(e)
            
        try:
            thinking = str(completion.content[0].thinking)
            output = str(completion.content[1].text)
            return {
                "thinking": thinking,
                "output": output
            }

        except Exception as e:
            return {
                "thinking": None,
                "output": None
            }