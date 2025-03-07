import os
from tqdm import tqdm
from typing import Union
import torch
from diffusers.utils import export_to_video

def run_cogvideox_2b(prompts:list,raw_video_dir:str,device_id:Union[int, list]=0,video_names:list=[],
                    num_frames:int=48,height:int=480,width:int=720,
                    num_inference_steps:int=50,guidance_scale:float=7.0,seed:int=42,
                    api_kwargs:dict={},model_dir:str="",script_dir:str="",): 
    
    torch.manual_seed(seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{device_id}"
    
    # from diffusers import CogVideoXPipeline
    from diffusers.pipelines.cogvideo.pipeline_cogvideox import CogVideoXPipeline
    
    pipe = CogVideoXPipeline.from_pretrained(
        "THUDM/CogVideoX-2b",
        torch_dtype=torch.float16
    )

    pipe.enable_model_cpu_offload()
    pipe.vae.enable_tiling()
    for idx,prompt in tqdm(enumerate(prompts)):
        video_path=os.path.join(raw_video_dir,f"{idx}.mp4")
        if video_names is not None and len(video_names)!=0:
            video_path=os.path.join(raw_video_dir,f"{video_names[idx]}.mp4")
        video_frames = pipe(
            prompt=prompt,
            num_videos_per_prompt=1,
            num_frames=num_frames,
            height=height,
            width=width, 
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=torch.Generator(device="cuda").manual_seed(seed),
        ).frames[0]
        export_to_video(video_frames,video_path)
        

def run_cogvideox_5b(prompts:list,raw_video_dir:str,device_id:Union[int, list]=0,video_names:list=[],
                    num_frames:int=40,height:int=480,width:int=720,
                    num_inference_steps:int=50,guidance_scale:float=7.0,seed:int=42,
                    api_kwargs:dict={},model_dir:str="",script_dir:str="",): 
    
    torch.manual_seed(seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{device_id}"
    
    # from diffusers import CogVideoXPipeline
    from diffusers.pipelines.cogvideo.pipeline_cogvideox import CogVideoXPipeline
    
    pipe = CogVideoXPipeline.from_pretrained(
        "THUDM/CogVideoX-5b",
        torch_dtype=torch.bfloat16
    )

    pipe.enable_model_cpu_offload()
    pipe.vae.enable_tiling()
    for idx,prompt in tqdm(enumerate(prompts)):
        video_path=os.path.join(raw_video_dir,f"{idx}.mp4")
        if video_names is not None and len(video_names)!=0:
            video_path=os.path.join(raw_video_dir,f"{video_names[idx]}.mp4")
        video_frames = pipe(
            prompt=prompt,
            num_frames=num_frames,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=torch.Generator(device="cuda").manual_seed(seed),
        ).frames[0]
        export_to_video(video_frames,video_path)
        

def run_cogvideox15_5b(prompts:list,raw_video_dir:str,device_id:Union[int, list]=0,video_names:list=[],
                    num_frames:int=40,height:int=768,width:int=1360,
                    num_inference_steps:int=50,guidance_scale:float=7.0,seed:int=42,
                    api_kwargs:dict={},model_dir:str="",script_dir:str="",): 
    torch.manual_seed(seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{device_id}"
    
    # from diffusers import CogVideoXPipeline
    from diffusers.pipelines.cogvideo.pipeline_cogvideox import CogVideoXPipeline
    
    pipe = CogVideoXPipeline.from_pretrained(
        "THUDM/CogVideoX1.5-5b",
        torch_dtype=torch.bfloat16
    )

    pipe.enable_model_cpu_offload()
    pipe.vae.enable_tiling()
    for idx,prompt in tqdm(enumerate(prompts)):
        video_path=os.path.join(raw_video_dir,f"{idx}.mp4")
        if video_names is not None and len(video_names)!=0:
            video_path=os.path.join(raw_video_dir,f"{video_names[idx]}.mp4")
        video_frames = pipe(
            prompt=prompt,
            num_videos_per_prompt=1,
            num_frames=num_frames,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=torch.Generator(device="cuda").manual_seed(seed),
        ).frames[0]
        export_to_video(video_frames,video_path)