import os
from tqdm import tqdm
from typing import Union
import torch
from diffusers.utils import export_to_video

def run_ltx_video_091(prompts:list,raw_video_dir:str,device_id:Union[int, list]=0,video_names:list=[],
                  num_frames:int=40,height:int=480,width:int=704,
                    num_inference_steps:int=50,guidance_scale:float=6.0,seed:int=42,
                    api_kwargs:dict={},model_dir:str="",script_dir:str="",): 
    
    torch.manual_seed(seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{device_id}"
    from diffusers.pipelines.ltx.pipeline_ltx import LTXPipeline
    
    pipe = LTXPipeline.from_pretrained("a-r-r-o-w/LTX-Video-0.9.1-diffusers", torch_dtype=torch.bfloat16)
    pipe.to("cuda")

    for idx,prompt in tqdm(enumerate(prompts)):
        video_path=os.path.join(raw_video_dir,f"{idx}.mp4")
        if video_names is not None and len(video_names)!=0:
            video_path=os.path.join(raw_video_dir,f"{video_names[idx]}.mp4")
        video_frames = pipe(
            prompt=prompt,
            num_frames=num_frames,
            width=width,
            height=height,
            decode_timestep=0.03,
            decode_noise_scale=0.025,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        ).frames[0]
        export_to_video(video_frames,video_path)