
import anthropic
import base64
import os
from tqdm import tqdm

def _refine_cmt_claude(model_name,model_access,comments,prompts,frames_2d_list,template,dim_def):
    os.environ["ANTHROPIC_API_KEY"]=model_access["api_key"]
    
    def claude_img_input(path_or_url):
        if "http" in path_or_url:
            return {
                        "type":"url",
                        "data":f"{path_or_url}"
                    }
        else:
            with open(path_or_url, "rb") as img_file:
                encoding=base64.standard_b64encode(img_file.read()).decode("utf-8")
            return {
                        "type":"base64",
                        "media_type":"image/jpeg",
                        "data":f"data:image/jpeg;base64,{encoding}"
                    }

    
    client = anthropic.Anthropic()
    final_list=[]
    
    # only text
    if len(frames_2d_list)==0 or frames_2d_list is None:
        for raw_comment,prompt in tqdm(zip(comments,prompts)):
            completion = client.messages.create(
                model="claude-opus-4-20250514",
                max_tokens=2000,
                temperature=0.7,
                system="You are an expert for evaluating and commenting on the quality of AI videos.",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": template.substitute(dim_def=dim_def,prompt=prompt,comment=raw_comment)
                            }
                        ]
                    }
                ]
            )
            try:
                res=completion["content"][0]["text"]
                res="{"+res.split("{")[-1].split("}")[0].strip()+"}"
                eval_res = eval(res)
                if isinstance(eval_res, dict) and 'comment' in eval_res:
                    final_list.append(eval_res["comment"])
            except Exception as e:
                final_list.append(" ")
    # text attached with images
    else:
        for raw_comment,prompt,frame_paths in tqdm(zip(comments,prompts,frames_2d_list)):  
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
                                "text": template.substitute(dim_def=dim_def,prompt=prompt,comment=raw_comment)
                                        +"\n\nHere is some frames of the video for your reference.",
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
                                for path in frame_paths
                            ],
                        ],
                    },
                ]
            )
            try:
                res=completion["content"][0]["text"]
                res="{"+res.split("{")[-1].split("}")[0].strip()+"}"
                eval_res = eval(res)
                if isinstance(eval_res, dict) and 'comment' in eval_res:
                    final_list.append(eval_res["comment"])
            except Exception as e:
                final_list.append(" ")
    
    
    return final_list