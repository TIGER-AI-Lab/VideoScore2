import os
from tqdm import tqdm
from typing import Union
import torch
from diffusers.utils import export_to_video
import subprocess
from datetime import datetime

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
        


def run_ltx_video_095(prompts:list,raw_video_dir:str,device_id:Union[int, list]=0,video_names:list=[],
                  num_frames:int=121,height:int=480,width:int=704,
                    num_inference_steps:int=40,guidance_scale:float=7.5,seed:int=42,
                    api_kwargs:dict={},model_dir:str="",script_dir:str="",): 
    
    os.chdir(f"{model_dir}/")
    date_time = datetime.now().strftime("%m-%d--%H-%M-%S")
    os.makedirs(f"{model_dir}/temp",exist_ok=True)
    
    prompt_file=f"temp/prompt_list_v1_3_{date_time}.txt"
    with open(os.path.join(model_dir,prompt_file),"w") as f:
        for item in prompts:
            f.write(repr(item)[1:-1] + '\n')
    
    video_names_file=f"temp/video_names.txt"
    with open(os.path.join(model_dir,video_names_file),"w") as f:
        for video_name in video_names:
            f.write(video_name+"\n")
    
    print("ready to run generating script")
    
    env = os.environ.copy() 
    env['CUDA_VISIBLE_DEVICES'] = f"{device_id}"
    subprocess.run([
        "python","inference_batch.py",
        "--ckpt_path",f"{os.path.join(model_dir,'ckpt/ltx-video-2b-v0.9.5.safetensors')}",
        "--prompt_file",f"{os.path.join(model_dir,prompt_file)}",
        "--video_names_file",f"{os.path.join(model_dir,video_names_file)}",
        "--output_path",f"{raw_video_dir}",
        "--height",f"{height}",
        "--width",f"{width}",
        "--num_frames",f"{num_frames}",
        "--seed",f"{seed}",
        "--num_inference_steps",f"{num_inference_steps}",
        "--guidance_scale",f"{guidance_scale}",
        
        ],env=env)
    
    os.chdir(script_dir)