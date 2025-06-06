from google import genai
from google.genai import types
from tqdm import tqdm
import requests

def _refine_cmt_gemini(model_name,model_access,comments,prompts,frames_2d_list,template,dim_def):  
    def gemini_img_input(path_or_url):
        if "http" in path_or_url:
            image_bytes = requests.get(path_or_url).content
        else:
            with open(path_or_url, "rb") as f:
                image_bytes = f.read()
        return types.Part.from_bytes(data=image_bytes,mime_type='image/jpeg',)    
    
    client = genai.Client(api_key=model_access["api_key"])
    final_list=[]
    
    # only text
    if len(frames_2d_list)==0 or frames_2d_list is None:
        for raw_comment,prompt in tqdm(zip(comments,prompts)):
            response = client.models.generate_content(
                model=model_name,
                contents=template.substitute(dim_def=dim_def,prompt=prompt,comment=raw_comment),
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
                    final_list.append(eval_res["comment"])
            except Exception as e:
                final_list.append(" ")
                
    # text attached with images
    else:
        for raw_comment,prompt,frame_paths in tqdm(zip(comments,prompts,frames_2d_list)):  
            response = client.models.generate_content(
                model=model_name,
                contents=[template.substitute(dim_def=dim_def,prompt=prompt,comment=raw_comment)
                                        +"\n\nHere is some frames of the video for your reference, If there is any discrepancy between the manual annotation and the video frames, please refer to the video frames as the standard.",
                
                ]+[
                    gemini_img_input(path) for path in frame_paths
                ],
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
                    final_list.append(eval_res["comment"])
            except Exception as e:
                final_list.append(" ")
    
    
    return final_list