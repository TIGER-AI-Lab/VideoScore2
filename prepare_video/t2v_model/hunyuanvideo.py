import os
import shutil
import subprocess
import yaml
from typing import Union
from datetime import datetime

hf_cache_dir="/data/shared_huggingface/hub/"

def run_hunyuanvideo_24g(prompts:list,raw_video_dir:str,device_id:Union[int, list]=0,video_names:list=[],
                   num_frames:int=97,height:int=544,width:int=960,
                   num_inference_steps:int=30,guidance_scale:float=6.0,seed:int=42,
                    api_kwargs:dict={},model_dir:str="",script_dir:str="",):
    
    
    os.chdir(f"{model_dir}/")
    os.makedirs(f"{model_dir}/temp",exist_ok=True)
    date_time = datetime.now().strftime("%m-%d--%H-%M-%S")
    prompt_file=f"{model_dir}/temp/hunyuanvideo_24g_prompt_list_{date_time}.txt"
    with open(prompt_file,"w") as f:
        for item in prompts:
            f.write(repr(item)[1:-1] + '\n')
    
    video_names_file=f"{model_dir}/temp/hunyuanvideo_24g_video_names_{date_time}.txt"
    with open(video_names_file,"w") as f:
        for video_name in video_names:
            f.write(video_name+"\n")
    
    print("ready to run generating script")
    
    env = os.environ.copy() 
    env['CUDA_VISIBLE_DEVICES'] = f"{device_id}"
        
    subprocess.run([
        "python","scripts/sample_hunyuanvideo_24g.py",
        "--prompt_file",f"{prompt_file}",
        "--video_names_file",f"{video_names_file}",
        "--num_frames",f"{num_frames}",
        "--num_inference_steps",f"{num_inference_steps}",
        "--guidance_scale",f"{guidance_scale}",
        "--width",f"{width}",
        "--height",f"{height}",
        "--seed",f"{seed}",
        "--save_dir",f"{raw_video_dir}",
        "--cache_dir",f"{hf_cache_dir}",
        ],env=env)
    
    os.chdir(script_dir)


def run_hunyuanvideo_80g(prompts:list,raw_video_dir:str,device_id:Union[int, list]=0,video_names:list=[],
                   num_frames:int=97,height:int=544,width:int=960,
                   num_inference_steps:int=30,guidance_scale:float=6.0,seed:int=42,
                    api_kwargs:dict={},model_dir:str="",script_dir:str="",):
    os.chdir(f"{model_dir}/")
    os.makedirs(f"{model_dir}/temp",exist_ok=True)
    date_time = datetime.now().strftime("%m-%d--%H-%M-%S")
    prompt_file=f"{model_dir}/temp/hunyuanvideo_80g_prompt_list_{date_time}.txt"
    with open(prompt_file,"w") as f:
        for item in prompts:
            f.write(repr(item)[1:-1] + '\n')
    
    video_names_file=f"{model_dir}/temp/hunyuanvideo_80g_video_names_{date_time}.txt"
    with open(video_names_file,"w") as f:
        for video_name in video_names:
            f.write(video_name+"\n")
    
    print("ready to run generating script")
    
    env = os.environ.copy() 
    env['CUDA_VISIBLE_DEVICES'] = f"{device_id}"

    subprocess.run([
        "python","scripts/sample_hunyuanvideo_80g.py",
        "--prompt_file",f"{prompt_file}",
        "--video_names_file",f"{video_names_file}",
        "--num_frames",f"{num_frames}",
        "--num_inference_steps",f"{num_inference_steps}",
        "--guidance_scale",f"{guidance_scale}",
        "--width",f"{width}",
        "--height",f"{height}",
        "--seed",f"{seed}",
        "--save_dir",f"{raw_video_dir}",
        "--cache_dir",f"{hf_cache_dir}",
        ],env=env)
    
    os.chdir(script_dir)
    
    
    