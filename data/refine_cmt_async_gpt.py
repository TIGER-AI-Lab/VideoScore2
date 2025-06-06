import os
import sys
import time
import base64
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