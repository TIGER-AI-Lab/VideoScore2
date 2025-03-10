import os
import shutil
import subprocess
from typing import Union
from datetime import datetime

def run_allegro(prompts:list,raw_video_dir:str,device_id:Union[int, list]=0,video_names:list=[],
                   num_frames:int=88,height:int=720,width:int=1280,
                   num_inference_steps:int=100,guidance_scale:float=7.5,seed:int=42,
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
        "python","single_inference.py",
        "--save_dir",f"{raw_video_dir}",
        "--prompt_file",f"{os.path.join(model_dir,prompt_file)}",
        "--video_names_file",f"{os.path.join(model_dir,video_names_file)}",    
        "--vae",f"{model_dir}/ckpt/Allegro/vae",
        "--dit",f"{model_dir}/ckpt/Allegro/transformer",
        "--text_encoder",f"{model_dir}/ckpt/Allegro/text_encoder",
        "--tokenizer",f"{model_dir}/ckpt/Allegro/tokenizer",
        "--guidance_scale",f"{guidance_scale}",
        "--num_sampling_steps",f"{num_inference_steps}",
        "--seed",f"{seed}",
        "--num_frames",f"{num_frames}",
        "--height",f"{height}",
        "--width",f"{width}",
        ],env=env)

    os.chdir(script_dir)