
import os
import subprocess
from typing import Union

def run_videocrafter2(prompts:list,raw_video_dir:str,device_id:Union[int, list]=0,video_names:list=[],
                   num_frames:int=24,height:int=320,width:int=512,
                   num_inference_steps:int=50,guidance_scale:float=12.0,seed:int=42,
                    api_kwargs:dict={},model_dir:str="",script_dir:str="",):
    
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{device_id}"
    
    prompt_file=f"{model_dir}/prompts/running_prompts.txt"
    video_names_file=f"{model_dir}/prompts/video_names.txt"
    ckpt_path=f"{model_dir}/checkpoints/base_512_v2/model.ckpt"
    config=f"{model_dir}/configs/inference_t2v_512_v2.0.yaml"
    
    with open(prompt_file,"w") as f:
        for prompt in prompts:
            f.write(prompt+"\n")
            
    with open(video_names_file,"w") as f:
        for video_name in video_names:
            f.write(video_name+"\n")

    os.chdir(f"{model_dir}/")
    print("ready to run generating script")
    subprocess.run(["python3", "scripts/evaluation/inference.py",
                    "--seed", f"{seed}",
                    "--mode", "base",
                    "--prompt_file", f"{prompt_file}",
                    "--video_names_file", f"{video_names_file}",
                    "--ckpt_path", f"{ckpt_path}",
                    "--config",f"{config}",
                    "--savedir", f"{raw_video_dir}",
                    "--n_samples", "1",
                    "--bs","1",
                    "--height", f"{height}",
                    "--width", f"{width}",
                    "--unconditional_guidance_scale",f"{guidance_scale}",
                    "--ddim_steps","50",
                    "--ddim_eta","1.0", 
                    "--fps","28",
                    ])
    print("current env: ",os.environ.get('CONDA_DEFAULT_ENV'))
    os.chdir(script_dir)





            
        
        


