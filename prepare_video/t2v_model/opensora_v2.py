import os
import subprocess
from typing import Union
import shutil
from datetime import datetime

def run_opensora_v2(prompts:list,raw_video_dir:str,device_id:Union[int, list]=0,video_names:list=[],
                   num_frames:int=97,height:int=480,width:int=270,
                   num_inference_steps:int=30,guidance_scale:float=7.5,seed:int=42,
                    api_kwargs:dict={},model_dir:str="",script_dir:str="",):
    
    
    os.chdir(f"{model_dir}/")
    date_time = datetime.now().strftime("%m-%d--%H-%M-%S")
    prompt_file=f"configs/opensora-v1-3/inference/prompt_{date_time}.txt"
    with open(os.path.join(model_dir,prompt_file),"w") as f:
        for item in prompts:
            f.write(repr(item)[1:-1] + '\n')
    
    video_names_file=f"configs/opensora-v1-3/inference/video_names_{date_time}.txt"
    with open(os.path.join(model_dir,video_names_file),"w") as f:
        for video_name in video_names:
            f.write(video_name+"\n")
    
    print("ready to run generating script")

    env = os.environ.copy() 
    env['PYTHONPATH'] = f"{model_dir}"
    env['CUDA_VISIBLE_DEVICES'] = f"{device_id}"    
    
    
    # torchrun --nproc_per_node 1 --standalone scripts/diffusion/inference.py configs/diffusion/inference/t2i2v_768px.py --save-dir samples --prompt "raining, sea"
    
    print(f"curr_seed: {seed}")
    subprocess.run(["torchrun", 
                "--nproc_per_node","1",
                "--standalone","scripts/diffusion/inference.py",
                "configs/diffusion/inference/t2i2v_768px.py",
                "--save-dir",f"{raw_video_dir}",
                
                # "--seed", f"{seed}",
                # "--num-frames",f"{num_frames}",
                # "--num-sample","1",
                # "--resolution","720p",
                # "--aspect-ratio","9:16",
                # "--aes","very good",
                # "--flow","fair",
                # "--save-dir",f"{raw_video_dir}",
                # "--prompt-path",f"{os.path.join(model_dir,prompt_file)}",
                # "--layernorm-kernel","False",
                # "--num-sampling-steps",f"{num_inference_steps}",
                # "--cfg-scale",f"{guidance_scale}",
                # "--video-names-path",f"{os.path.join(model_dir,video_names_file)}",
                ],env=env)
    
    os.chdir(script_dir)
    
    # for video in os.listdir(raw_video_dir):
    #     input_video=os.path.join(raw_video_dir,video)
    #     output_video=os.path.join(raw_video_dir,video)
    #     cmd = [
    #         "ffmpeg",
    #         "-i", input_video,
    #         "-vcodec", "libx264",
    #         output_video
    #     ]
    #     subprocess.run(cmd, check=True)