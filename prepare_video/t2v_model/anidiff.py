import os
from tqdm import tqdm
from typing import Union
import torch
from diffusers.utils import export_to_video

def run_anidiff(prompts:list,raw_video_dir:str,device_id:Union[int, list]=0,video_names:list=[],
                num_frames:int=16,height:int=512,width:int=512,
                    num_inference_steps:int=50,guidance_scale:float=7.5,seed:int=42,
                    api_kwargs:dict={},model_dir:str="",script_dir:str="",):   
    
    torch.manual_seed(seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{device_id}"
    
    # from diffusers import AnimateDiffPipeline, DDIMScheduler, MotionAdapter
    from diffusers.pipelines.animatediff.pipeline_animatediff import AnimateDiffPipeline
    from diffusers.schedulers.scheduling_ddim import DDIMScheduler
    from diffusers.models.unets.unet_motion_model import MotionAdapter

    # Load the motion adapter
    adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2", torch_dtype=torch.float16)
    # load SD 1.5 based finetuned model
    model_id = "SG161222/Realistic_Vision_V5.1_noVAE"
    pipe = AnimateDiffPipeline.from_pretrained(model_id, motion_adapter=adapter, torch_dtype=torch.float16)
    scheduler = DDIMScheduler.from_pretrained(
        model_id,
        subfolder="scheduler",
        clip_sample=False,
        timestep_spacing="linspace",
        beta_schedule="linear",
        steps_offset=1,
    )
    pipe.scheduler = scheduler

    # enable memory savings
    pipe.enable_vae_slicing()
    pipe.enable_model_cpu_offload()
    
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
        ).frames[0]
        export_to_video(video_frames,video_path)
        
  
def run_anidiff_sdxl(prompts:list,raw_video_dir:str,device_id:Union[int, list]=0,video_names:list=[],
                    num_frames:int=16,height:int=512,width:int=512,
                    num_inference_steps:int=50,guidance_scale:float=5.0,seed:int=42,
                    api_kwargs:dict={},model_dir:str="",script_dir:str="",): 
    
    torch.manual_seed(seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{device_id}"
    
    from diffusers.schedulers.scheduling_ddim import DDIMScheduler
    from diffusers.models.unets.unet_motion_model import MotionAdapter
    from diffusers.pipelines.animatediff.pipeline_animatediff_sdxl import AnimateDiffSDXLPipeline
    
    adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-sdxl-beta", torch_dtype=torch.float16)
    model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    scheduler = DDIMScheduler.from_pretrained(
        model_id, subfolder="scheduler", clip_sample=False, timestep_spacing="linspace", beta_schedule="linear", steps_offset=1,
    )
    pipe = AnimateDiffSDXLPipeline.from_pretrained(
        model_id, motion_adapter=adapter, scheduler=scheduler, torch_dtype=torch.float16, variant="fp16",
    ).to("cuda")

    # enable memory savings
    pipe.enable_vae_slicing()
    pipe.enable_vae_tiling()

    for idx,prompt in tqdm(enumerate(prompts)):
        video_path=os.path.join(raw_video_dir,f"{idx}.mp4")
        if video_names is not None and len(video_names)!=0:
            video_path=os.path.join(raw_video_dir,f"{video_names[idx]}.mp4")
        video_frames = pipe(
            prompt=prompt,
            num_frames=num_frames,
            height=height,
            width=width, 
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            # generator=torch.Generator(device="cuda").manual_seed(seed),
        ).frames[0]
        export_to_video(video_frames,video_path)