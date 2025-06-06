import base64
from openai import OpenAI
from tqdm import tqdm



def _refine_cmt_gpt(model_name,model_access,comments,prompts,frames_2d_list,template,dim_def):
    def get_img_url(path_or_url):
        if "http" in path_or_url:
            return path_or_url
        else:
            with open(path_or_url, "rb") as img_file:
                encoding=base64.b64encode(img_file.read()).decode("utf-8")
        return f"data:image/jpeg;base64,{encoding}"
        
    
    client = OpenAI(api_key=model_access["api_key"], base_url=model_access["base_url"])
    final_list=[]
    
    # only text
    if len(frames_2d_list)==0 or frames_2d_list is None:
        for raw_comment,prompt in tqdm(zip(comments,prompts)):
            completion = client.chat.completions.create(
                model=model_name,
                temperature=0.7,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": template.substitute(dim_def=dim_def,prompt=prompt,comment=raw_comment)}
                        ]
                    }
                ]
            )
            try:
                res=completion.choices[0].message["content"]
                res="{"+res.split("{")[-1].split("}")[0].strip()+"}"
                eval_res = eval(res)
                if isinstance(eval_res, dict) and 'comment' in eval_res:
                    final_list.append(eval_res["comment"])
            except Exception as e:
                final_list.append(" ")
                
    # text attached with images
    else:
        for raw_comment,prompt,frames_path in tqdm(zip(comments,prompts,frames_2d_list)):  
            completion = client.chat.completions.create(
                model=model_name,
                temperature=0.7,
                messages=[
                    {
                        "role":"user",
                        "content": [
                            {
                                "type": "text", 
                                "text": template.substitute(dim_def=dim_def,prompt=prompt,comment=raw_comment)
                                        +"\n\nHere is some frames of the video for your reference, ",
                            },
                            *[
                                {
                                    "type": "image_url",
                                    "image_url": {"url": get_img_url(path_or_url)}
                                }  for path_or_url in frames_path
                            ]
                        ],
                    },
                ]
            )
            try:
                res=completion.choices[0].message["content"]
                res="{"+res.split("{")[-1].split("}")[0].strip()+"}"
                eval_res = eval(res)
                if isinstance(eval_res, dict) and 'comment' in eval_res:
                    final_list.append(eval_res["comment"])
            except Exception as e:
                final_list.append(" ")
    
    
    return final_list
    
    
    