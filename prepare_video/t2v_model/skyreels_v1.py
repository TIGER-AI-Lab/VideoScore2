import os
import shutil
import subprocess
import yaml
from typing import Union
from datetime import datetime

def run_skyreels_v1(prompts:list,raw_video_dir:str,device_id:Union[int, list]=0,video_names:list=[],
                   num_frames:int=97,height:int=544,width:int=960,
                   num_inference_steps:int=30,guidance_scale:float=6.0,seed:int=42,
                    api_kwargs:dict={},model_dir:str="",script_dir:str="",):
    os.chdir(f"{model_dir}/")
    os.makedirs(f"{model_dir}/temp",exist_ok=True)
    date_time = datetime.now().strftime("%m-%d--%H-%M-%S")
    prompt_file=f"temp/prompt_list_{date_time}.txt"
    with open(os.path.join(model_dir,prompt_file),"w") as f:
        for item in prompts:
            f.write(repr(item)[1:-1] + '\n')
    
    video_names_file=f"temp/video_names_{date_time}.txt"
    with open(os.path.join(model_dir,video_names_file),"w") as f:
        for video_name in video_names:
            f.write(video_name+"\n")
    
    print("ready to run generating script")
    
    env = os.environ.copy() 
    env['CUDA_VISIBLE_DEVICES'] = f"{device_id}"
    
    subprocess.run([
        "python3","video_generate.py",
        "--model_id","Skywork/SkyReels-V1-Hunyuan-T2V",
        "--task_type","t2v",
        "--out_dir",f"{raw_video_dir}",
        "--guidance_scale","6.0",
        "--height",f"{height}",
        "--width",f"{width}",
        "--num_frames",f"{num_frames}",
        "--prompt_file",f"{os.path.join(model_dir,prompt_file)}",
        "--embedded_guidance_scale","1.0",
        "--quant",
        "--offload",
        "--high_cpu_memory",
        "--parameters_level",
        "--video_names_file",f"{os.path.join(model_dir,video_names_file)}",
        ],env=env)

    os.chdir(script_dir)