import os
import shutil
import subprocess
import yaml
from typing import Union
from datetime import datetime


def run_stepvideo_t2v(prompts:list,raw_video_dir:str,device_id:Union[int, list]=0,video_names:list=[],
                   num_frames:int=97,height:int=544,width:int=960,
                   num_inference_steps:int=30,guidance_scale:float=6.0,seed:int=42,
                    api_kwargs:dict={},model_dir:str="",script_dir:str="",):
    os.chdir(f"{model_dir}/")
    os.makedirs(f"{model_dir}/temp",exist_ok=True)
    date_time = datetime.now().strftime("%m-%d--%H-%M-%S")
    prompt_file=f"{model_dir}/temp/stepvideo_t2v_prompt_list_{date_time}.txt"
    with open(prompt_file,"w") as f:
        for item in prompts:
            f.write(repr(item)[1:-1] + '\n')
    
    video_names_file=f"{model_dir}/temp/stepvideo_t2v_video_names_{date_time}.txt"
    with open(video_names_file,"w") as f:
        for video_name in video_names:
            f.write(video_name+"\n")
    
    print("ready to run generating script")
    
    env = os.environ.copy() 
    env['CUDA_VISIBLE_DEVICES'] = f"{device_id}"
        
    subprocess.run([
        "python","sample_stepvideo.py",
        "--prompt_file",f"{prompt_file}",
        "--video_names_file",f"{video_names_file}",
        "--num_frames",f"{num_frames}",
        "--num_inference_steps",f"{num_inference_steps}",
        "--guidance_scale",f"{guidance_scale}",
        "--width",f"{width}",
        "--height",f"{height}",
        "--seed",f"{seed}",
        "--save_dir",f"{raw_video_dir}",
        ],env=env)
    
    os.chdir(script_dir)


def run_stepvideo_t2v_low_vram(prompts:list,raw_video_dir:str,device_id:Union[int, list]=0,video_names:list=[],
                   num_frames:int=97,height:int=544,width:int=960,
                   num_inference_steps:int=30,guidance_scale:float=6.0,seed:int=42,
                    api_kwargs:dict={},model_dir:str="",script_dir:str="",):
    os.chdir(f"{model_dir}/")
    os.makedirs(f"{model_dir}/temp",exist_ok=True)
    date_time = datetime.now().strftime("%m-%d--%H-%M-%S")
    prompt_file=f"{model_dir}/temp/stepvideo_t2v_low_vram_prompt_list_{date_time}.txt"
    with open(prompt_file,"w") as f:
        for item in prompts:
            f.write(repr(item)[1:-1] + '\n')
    
    video_names_file=f"{model_dir}/temp/stepvideo_t2v_low_vram_video_names_{date_time}.txt"
    with open(video_names_file,"w") as f:
        for video_name in video_names:
            f.write(video_name+"\n")
    
    print("ready to run generating script")
    
    env = os.environ.copy() 
    env['CUDA_VISIBLE_DEVICES'] = f"{device_id}"

    subprocess.run([
        "python","scripts/sample_stepvideo_low_vram.py",
        "--prompt_file",f"{prompt_file}",
        "--video_names_file",f"{video_names_file}",
        "--num_frames",f"{num_frames}",
        "--num_inference_steps",f"{num_inference_steps}",
        "--guidance_scale",f"{guidance_scale}",
        "--width",f"{width}",
        "--height",f"{height}",
        "--seed",f"{seed}",
        "--save_dir",f"{raw_video_dir}",
        ],env=env)
    
    os.chdir(script_dir)
    
    
    