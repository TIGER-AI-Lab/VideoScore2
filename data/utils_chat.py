import base64
import os
from PIL import Image
import io
import warnings
import time
import random
import anthropic
import httpx
import json
import requests

def _thinking_cmt_claude_few_shot(
    model_access,
    template,
    sys_prompt,
    dim_defs,
    t2v_prompt,
    raw_comments,
    frame_list,
    few_shot_examples,
    shot_num)->dict:

    s_t=time.time()
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
        elif isinstance(path_url_pil, (bytes, bytearray)):
            base64_str = base64.b64encode(path_url_pil).decode("utf-8")
            return base64_str
        
        else:
            warnings.warn("image input is not url, path or PIL Image!")
            return ""
    
    client = anthropic.Anthropic()
    
    if few_shot_examples is not None and len(few_shot_examples)>0:
        few_shot_messages = list()
        if len(few_shot_examples) < shot_num:
            warnings.warn(f"few_shot_examples length {len(few_shot_examples)} is less than shot_num {shot_num}, using all examples.")
            shot_num = len(few_shot_examples)
        for example in random.sample(few_shot_examples, k = min(shot_num, len(few_shot_examples))):
            base64_str_list= example["frame_base64_list"]
            if "raw_comments" in example and len(example["raw_comments"])==3:
                few_shot_messages.extend(
                    [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": template.substitute(visual_def=dim_defs[0],t2v_def=dim_defs[1],phy_def=dim_defs[2],
                                            comment_visual=example["raw_comments"][0], comment_t2v=example["raw_comments"][1],
                                            comment_phy=example["raw_comments"][2], prompt=t2v_prompt)
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
                                    for base64_str in base64_str_list
                                ]
                            ],
                        },
                        {
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "thinking",
                                    "thinking": example["thinking"],
                                    "signature":"",
                                },
                                {
                                    "type": "text",
                                    "text":json.dumps(example["output"], indent=4)    
                                }
                            ]
                        }
                    ]
                )
        # print(len(few_shot_messages[0]['content'][1]['source']['data']))
        # print(few_shot_messages[1]['content'][0]['thinking'])
        # print(few_shot_messages[1]['content'][1]['text'])
    
    base64_str_list = [claude_img_input(path) for path in frame_list]
    try:
        input_text=template.substitute(visual_def=dim_defs[0],t2v_def=dim_defs[1],phy_def=dim_defs[2],
                                                comment_visual=raw_comments[0],
                                                comment_t2v=raw_comments[1],
                                                comment_phy=raw_comments[2],
                                                prompt=t2v_prompt)
        if frame_list is not None and len(frame_list)>0:
            input_text+="\n\nHere is some frames of the video for your reference, If there is any discrepancy between the manual annotation and the video frames, please refer to the video frames as the standard."
        
        completion = client.messages.create(
            model=model_name,
            max_tokens=8000,
            thinking={
                "type": "enabled",
                "budget_tokens": 5000
            },
            system=sys_prompt,
            messages=list(few_shot_messages)+[
            # messages=[
                {
                    "role":"user",
                    "content": [
                        {
                            "type": "text", 
                            "text": input_text,
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

        thinking = str(completion.content[0].thinking)
        output = str(completion.content[1].text)
        return {
            "thinking": thinking,
            "output": output
        }

    except Exception as e:
        print(e)
        return {
            "thinking": None,
            "output": None
        }
        
        

def _thinking_cmt_claude_few_shot_OR(
    model_access,
    template,
    sys_prompt,
    dim_defs,
    t2v_prompt,
    raw_comments,
    frame_list,
    few_shot_examples,
    shot_num)->dict:
    
    from openai import OpenAI
    
    base_url=model_access["base_url"]
    api_key=model_access["api_key"]
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
    
    if few_shot_examples is not None and len(few_shot_examples)>0:
        few_shot_messages = list()
        if len(few_shot_examples) < shot_num:
            warnings.warn(f"few_shot_examples length {len(few_shot_examples)} is less than shot_num {shot_num}, using all examples.")
            shot_num = len(few_shot_examples)
        for example in random.sample(few_shot_examples, k = min(shot_num, len(few_shot_examples))):
            base64_str_list= example["frame_base64_list"]
            if "raw_comments" in example and len(example["raw_comments"])==3:
                few_shot_messages.extend(
                    [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": template.substitute(visual_def=dim_defs[0],t2v_def=dim_defs[1],phy_def=dim_defs[2],
                                            comment_visual=example["raw_comments"][0], comment_t2v=example["raw_comments"][1],
                                            comment_phy=example["raw_comments"][2], prompt=t2v_prompt)
                                },
                                *[
                                    {
                                        "type": "image_url",
                                        "image_url":{
                                            "url":f"data:image/jpeg;base64,{base64_str}"
                                        }
                                    }  
                                    for base64_str in base64_str_list
                                ]
                            ],
                        },
                        {
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "text",
                                    "text":json.dumps(example["output"], indent=4)    
                                }
                            ],
                            "reasoning":[
                                {
                                    "type": "text",
                                    "text":example["thinking"],
                                }
                            ]
                        }
                    ]
                )
        # print(len(few_shot_messages[0]['content'][1]['source']['data']))
        # print(few_shot_messages[1]['content'][0]['thinking'])
        # print(few_shot_messages[1]['content'][1]['text'])
    
    base64_str_list = [claude_img_input(path) for path in frame_list]
    try:
        input_text=template.substitute(visual_def=dim_defs[0],t2v_def=dim_defs[1],phy_def=dim_defs[2],
                                                comment_visual=raw_comments[0],
                                                comment_t2v=raw_comments[1],
                                                comment_phy=raw_comments[2],
                                                prompt=t2v_prompt)
        if frame_list is not None and len(frame_list)>0:
            input_text+="\n\nHere is some frames of the video for your reference, If there is any discrepancy between the manual annotation and the video frames, please refer to the video frames as the standard."
        
        url = f"{base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload={
            "extra_headers":{},
            "extra_body":{},
            "model":model_name,
            "messages":list(few_shot_messages)+[
            # messages=[
                {
                    "role":"system",
                    "content": [
                        {
                        "type": "text",
                        "text": sys_prompt,
                        },
                    ]
                },
                {
                    "role":"user",
                    "content": [
                        {
                            "type": "text", 
                            "text": input_text,
                        },
                        *[
                            {
                                "type": "image_url",
                                "image_url":{
                                    "url":f"data:image/jpeg;base64,{base64_str}"
                                }
                            }  
                            for _,base64_str in zip(frame_list, base64_str_list)
                        ]
                    ],
                },
            ],
            "reasoning": {
                # "effort": "high",
                "max_tokens": 5000,
                "exclude": False, 
                "enabled": True 
            }
        }
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        thinking = str(response.json()['choices'][0]['message']['reasoning'])
        output = str(response.json()['choices'][0]['message']['content'])
        print(response.json()['choices'][0]['message'])
        print(len(thinking))
        exit()
        return {
            "thinking": thinking,
            "output": output
        }

    except Exception as e:
        print(e)
        return {
            "thinking": None,
            "output": None
        }