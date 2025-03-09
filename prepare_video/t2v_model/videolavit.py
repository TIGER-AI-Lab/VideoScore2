
import os
import subprocess
from typing import Union
from datetime import datetime

def run_videolavit(prompts:list,raw_video_dir:str,device_id:Union[int, list]=0,video_names:list=[],
                   num_frames:int=40,height:int=768,width:int=768,
                   num_inference_steps:int=50,guidance_scale:float=7.0,seed:int=42,
                    api_kwargs:dict={},model_dir:str="",script_dir:str="",):
    date_time = datetime.now().strftime("%m-%d--%H-%M-%S")
    
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{device_id}"
    prompt_file=f"{model_dir}/assets/prompts_{date_time}.txt"
    video_names_file=f"{model_dir}/assets/video_names_{date_time}.txt"
    
    with open(video_names_file,"w") as f:
        for video_name in video_names:
            f.write(video_name+"\n")
    
    with open(prompt_file,"w") as f:
        for prompt in prompts:
            f.write(prompt+"\n")

    os.chdir(f"{model_dir}/")
    print("ready to run generating script")
    subprocess.run(["python", "t2v_inference.py",
                    "--prompt_file", f"{prompt_file}",
                    "--video_names_file",f"{video_names_file}",
                    "--save_dir", f"{raw_video_dir}",
                    "--guidance_scale",f"{guidance_scale}",
                    "--num_inference_steps",f"{num_inference_steps}",
                    "--seed",f"{seed}",
                    ])
    print("current env: ",os.environ.get('CONDA_DEFAULT_ENV'))
    os.chdir(script_dir)





            
        
        




