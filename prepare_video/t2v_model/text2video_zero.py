import os
from tqdm import tqdm
from typing import Union
import torch
from diffusers.utils import export_to_video

def run_text2video_zero(prompts:list,raw_video_dir:str,device_id:Union[int, list]=0,video_names:list=[],
                  num_frames:int=8,height:int=256,width:int=256,
                    num_inference_steps:int=50,guidance_scale:float=7.5,seed:int=42,
                    api_kwargs:dict={},model_dir:str="",script_dir:str="",): 
    
    torch.manual_seed(seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{device_id}"
    
    from diffusers import TextToVideoZeroPipeline
    import imageio
    
    model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    pipe = TextToVideoZeroPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")    
    
    for idx,prompt in tqdm(enumerate(prompts)):
        video_path=os.path.join(raw_video_dir,f"{idx}.mp4")
        if video_names is not None and len(video_names)!=0:
            video_path=os.path.join(raw_video_dir,f"{video_names[idx]}.mp4")
        result = pipe(
            prompt=prompt,
            video_length=num_frames,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            ).images
        result = [(r * 255).astype("uint8") for r in result]
        imageio.mimsave(video_path, result)