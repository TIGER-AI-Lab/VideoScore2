import base64
import os
import requests
from PIL import Image
import io
import warnings


def _refine_cmt_gpt(model_name,model_access,score,raw_comment,prompt,frame_list,template,dim_def):
    from openai import OpenAI
    def gpt_img_input(path_url_pil):
        if isinstance(path_url_pil,str) and "http" in path_url_pil:
            return path_url_pil
        elif isinstance(path_url_pil,str) and not "http" in path_url_pil:
            with open(path_url_pil, "rb") as img_file:
                base64_str=base64.standard_b64encode(img_file.read()).decode("utf-8")
            return f"data:image/jpeg;base64,{base64_str}"
        elif isinstance(path_url_pil,Image.Image):
            buffered = io.BytesIO()
            path_url_pil.save(buffered, format=path_url_pil.format)
            base64_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            return f"data:image/jpeg;base64,{base64_str}"
        else:
            warnings.warn("image input is not url, path or PIL Image!")
            return f"data:image/jpeg;base64,{None}"
        
    client = OpenAI(api_key=model_access["api_key"], base_url=model_access["base_url"])
    
    # only text
    if len(frame_list)==0 or frame_list is None:
        completion = client.chat.completions.create(
            model=model_name,
            temperature=0.7,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", 
                         "text": template.substitute(dim_def=dim_def,score=score,comment=raw_comment,prompt=prompt),
                        }
                    ]
                }
            ]
        )
        try:
            res=completion.choices[0].message.content
            print(res)
            res="{"+res.split("{")[-1].split("}")[0].strip()+"}"
            eval_res = eval(res)
            if isinstance(eval_res, dict) and 'comment' in eval_res:
                return eval_res["comment"]
        except Exception as e:
            print(e)
            return None
                
    # text attached with images
    else:
        if "4o" in model_name:
            frame_list=frame_list[:3]
        completion = client.chat.completions.create(
            model=model_name,
            temperature=0.7,
            messages=[
                {
                    "role":"user",
                    "content": [
                        {
                            "type": "text", 
                            "text": template.substitute(dim_def=dim_def,score=score,comment=raw_comment,prompt=prompt,)
                                    +"\n\nHere is some frames of the video for your reference, If there is any discrepancy between the manual annotation and the video frames, please refer to the video frames as the standard.",
                        },
                        *[
                            {
                                "type": "image_url",
                                "image_url": {"url": gpt_img_input(path_or_url)}
                            }  for path_or_url in frame_list
                        ]
                    ],
                },
            ]
        )
        try:
            res=completion.choices[0].message.content
            print(res)
            res="{"+res.split("{")[-1].split("}")[0].strip()+"}"
            eval_res = eval(res)
            if isinstance(eval_res, dict) and 'comment' in eval_res:
                return eval_res["comment"]
        except Exception as e:
            print(e)
            return None
    
    




def _refine_cmt_gemini(model_name,model_access,score,raw_comment,prompt,frame_list,template,dim_def):  
    from google import genai
    from google.genai import types

    def gemini_img_input(path_url_pil):
        if isinstance(path_url_pil,str) and "http" in path_url_pil:
            image_bytes = requests.get(path_url_pil).content
        elif isinstance(path_url_pil,str) and not "http" in path_url_pil:
            with open(path_url_pil, "rb") as f:
                image_bytes = f.read()
        elif isinstance(path_url_pil,Image.Image):
            buffer = io.BytesIO()
            path_url_pil.save(buffer, format=path_url_pil.format) 
            image_bytes = buffer.getvalue()
        else:
            warnings.warn("image input is not url, path or PIL Image!")
            image_types=None    
        return types.Part.from_bytes(data=image_bytes,mime_type='image/jpeg',)    
    
    client = genai.Client(api_key=model_access["api_key"])
    
    # only text
    if len(frame_list)==0 or frame_list is None:
        response = client.models.generate_content(
            model=model_name,
            contents=template.substitute(dim_def=dim_def,score=score,comment=raw_comment,prompt=prompt),
            config=types.GenerateContentConfig(
                system_instruction="You are an expert for evaluating and commenting on the quality of AI videos.",
                temperature=0.7,
            )
        )
        try:
            res=response.text
            res="{"+res.split("{")[-1].split("}")[0].strip()+"}"
            eval_res = eval(res)
            if isinstance(eval_res, dict) and 'comment' in eval_res:
                return eval_res["comment"]
        except Exception as e:
            return None
                
    # text attached with images
    else:
        response = client.models.generate_content(
            model=model_name,
            contents=[template.substitute(dim_def=dim_def,score=score,comment=raw_comment,prompt=prompt)
                                    +"\n\nHere is some frames of the video for your reference, If there is any discrepancy between the manual annotation and the video frames, please refer to the video frames as the standard.",
            
            ]+[gemini_img_input(path) for path in frame_list],
            config=types.GenerateContentConfig(
                system_instruction="You are an expert for evaluating and commenting on the quality of AI videos.",
                temperature=0.7,
            )
        )
        try:
            res=response.text
            res="{"+res.split("{")[-1].split("}")[0].strip()+"}"
            eval_res = eval(res)
            if isinstance(eval_res, dict) and 'comment' in eval_res:
                return eval_res["comment"]
        except Exception as e:
            return None



def _refine_cmt_claude(model_name,model_access,score,raw_comment,prompt,frame_list,template,dim_def):
    import anthropic
    os.environ["ANTHROPIC_API_KEY"]=model_access["api_key"]

    def claude_img_input(path_url_pil):
        if isinstance(path_url_pil,str) and "http" in path_url_pil:
            return {
                        "type":"url","data":f"{path_url_pil}"
                    }
        elif isinstance(path_url_pil,str) and not "http" in path_url_pil:
            with open(path_url_pil, "rb") as img_file:
                base64_str=base64.standard_b64encode(img_file.read()).decode("utf-8")
            return {
                        "type":"base64",  "media_type":"image/jpeg",
                        "data":f"data:image/jpeg;base64,{base64_str}"
                    }
        elif isinstance(path_url_pil,Image.Image):
            buffered = io.BytesIO()
            path_url_pil.save(buffered, format=path_url_pil.format)
            base64_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            return {
                        "type":"base64",  "media_type":"image/jpeg",
                        "data":f"data:image/jpeg;base64,{base64_str}"
                    }
        else:
            warnings.warn("image input is not url, path or PIL Image!")
            return {"type":"url","data":None}
    
    client = anthropic.Anthropic()
    
    # only text
    if len(frame_list)==0 or frame_list is None:
        completion = client.messages.create(
            model=model_name,
            max_tokens=2000,
            temperature=0.7,
            system="You are an expert for evaluating and commenting on the quality of AI videos.",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": template.substitute(dim_def=dim_def,score=score,comment=raw_comment,prompt=prompt)
                        }
                    ]
                }
            ]
        )
        try:
            res=completion.content[0].text
            res="{"+res.split("{")[-1].split("}")[0].strip()+"}"
            eval_res = eval(res)
            if isinstance(eval_res, dict) and 'comment' in eval_res:
                return eval_res["comment"]
        except Exception as e:
            return None
    # text attached with images
    else:
        completion = client.messages.create(
            model=model_name,
            max_tokens=2000,
            temperature=0.7,
            system="You are an expert for evaluating and commenting on the quality of AI videos.",
            messages=[
                {
                    "role":"user",
                    "content": [
                        {
                            "type": "text", 
                            "text": template.substitute(dim_def=dim_def,score=score,comment=raw_comment,prompt=prompt)
                                    +"\n\nHere is some frames of the video for your reference, If there is any discrepancy between the manual annotation and the video frames, please refer to the video frames as the standard.",
                        },
                        *[
                            {
                                "type": "image",
                                "source":{
                                    "type":"base64",
                                    "media_type":"image/jpeg",
                                    "data":f"data:image/jpeg;base64,{claude_img_input(path)}"
                                },
                            }  
                            for path in frame_list
                        ],
                    ],
                },
            ]
        )
        try:
            res=completion.content[0].text
            res="{"+res.split("{")[-1].split("}")[0].strip()+"}"
            eval_res = eval(res)
            if isinstance(eval_res, dict) and 'comment' in eval_res:
                return eval_res["comment"]
        except Exception as e:
            return None



# ==================== discarded ====================
import sys
import time
from zeno_build.models import lm_config
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
    # print(sys.path)
from prepare_video.prompts.utils_gpt_chat import *

async def _refine_cmt_async_gpt(model_name,model_access,comments,prompts,frames_2d_list,template,dim_def):
    
    def encode_image(image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")
    
    model_config = lm_config.LMConfig(provider="openai_chat", model=model_name)
    os.environ["OPENAI_API_KEY"]=model_access["api_key"]
    os.environ["OPENAI_BASE_URL"]=model_access["base_url"]
    
    date_time=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    os.makedirs(f"./logs",exist_ok=True)
    logger=set_logger(f"./logs/import_anno_{date_time}.log")
    
    # only text
    if len(frames_2d_list)==0 or frames_2d_list is None:
        context_list=[]
        for raw_comment,prompt in zip(comments,prompts):
            context=dict(messages=
                [{
                    "content":template.substitute(dim_def=dim_def,prompt=prompt,comment=raw_comment),
                    "role":"user"
                }])
            context_list.append(context)
    
    # text attached with images
    else:
        context_list=[]
        
        for raw_comment,prompt,frame_paths in zip(comments,prompts,frames_2d_list):
            context=dict(messages=
                [{
                    "content": [
                        {
                            "type": "text", 
                            "text": template.substitute(dim_def=dim_def,prompt=prompt,comment=raw_comment)
                                        +"\n\nHere is some frames of the video for your reference.",
                        },
                        *[
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{encode_image(path)}"}
                            }  for path in frame_paths
                        ],
                    ],  
                    "role":"user"
                }])
            context_list.append(context)
                
            
    res_list = await generate_from_openai_chat_completion(
        full_contexts = context_list,
        model_config = model_config,
        logger=logger,
    )
    final_list=[]
    for res in res_list:
        try:
            res="{"+res.split("{")[-1].split("}")[0].strip()+"}"
            eval_res = eval(res)
            if isinstance(eval_res, dict) and 'comment' in eval_res:
                final_list.append(eval_res["comment"])
        except Exception as e:
            final_list.append(" ")
    return final_list
# ==================== discarded ====================