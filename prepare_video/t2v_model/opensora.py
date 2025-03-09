import os
import subprocess
from typing import Union
import shutil
from datetime import datetime

def run_opensora_v1_3(prompts:list,raw_video_dir:str,device_id:Union[int, list]=0,video_names:list=[],
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

    print(f"curr_seed: {seed}")
    subprocess.run(["python", "scripts/inference.py","configs/opensora-v1-3/inference/t2v.py",
                "--seed", f"{seed}",
                "--num-frames",f"{num_frames}",
                "--num-sample","1",
                "--resolution","720p",
                "--aspect-ratio","9:16",
                "--aes","very good",
                "--flow","fair",
                "--save-dir",f"{raw_video_dir}",
                "--prompt-path",f"{os.path.join(model_dir,prompt_file)}",
                "--layernorm-kernel","False",
                "--num-sampling-steps",f"{num_inference_steps}",
                "--cfg-scale",f"{guidance_scale}",
                "--video-names-path",f"{os.path.join(model_dir,video_names_file)}",
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


def run_opensora_v1_2(prompts:list,raw_video_dir:str,device_id:Union[int, list]=0,video_names:list=[],
                   num_frames:int=49,height:int=480,width:int=270,
                   num_inference_steps:int=30,guidance_scale:float=7.5,seed:int=42,
                    api_kwargs:dict={},model_dir:str="",script_dir:str="",):
    os.chdir(f"{model_dir}/")
    date_time = datetime.now().strftime("%m-%d--%H-%M-%S")
    prompt_file=f"configs/opensora-v1-2/inference/prompt_{date_time}.txt"
    with open(os.path.join(model_dir,prompt_file),"w") as f:
        for item in prompts:
            f.write(repr(item)[1:-1] + '\n')
    
    video_names_file=f"configs/opensora-v1-2/inference/video_names_{date_time}.txt"
    with open(os.path.join(model_dir,video_names_file),"w") as f:
        for video_name in video_names:
            f.write(video_name+"\n")
            
    print("ready to run generating script")

    env = os.environ.copy() 
    env['PYTHONPATH'] = f"{model_dir}"
    env['CUDA_VISIBLE_DEVICES'] = f"{device_id}"
    
    # date_time = datetime.now().strftime("%m-%d--%H-%M-%S")
    # # temp_save_dir=f"{model_dir}/res_v1_2_{date_time}"
    # temp_save_dir=f"{model_dir}/res_v1_2"
    # os.makedirs(temp_save_dir,exist_ok=True)

    print(f"curr_seed: {seed}")
    subprocess.run(["python", "scripts/inference.py","configs/opensora-v1-2/inference/sample.py",
                "--seed", f"{seed}",
                "--num-frames",f"{num_frames}",
                "--resolution","480p",
                "--aspect-ratio","9:16",
                "--save-dir",f"{raw_video_dir}",
                "--prompt-path",f"{os.path.join(model_dir,prompt_file)}",
                "--layernorm-kernel","False",
                "--num-sampling-steps",f"{num_inference_steps}",
                "--cfg-scale",f"{guidance_scale}",
                "--video-names-path",f"{os.path.join(model_dir,video_names_file)}",
                ],env=env)

    
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
    
    # os.chdir(script_dir)
    
    
def run_opensora_v1_1(prompts:list,raw_video_dir:str,device_id:Union[int, list]=0,video_names:list=[],
                   num_frames:int=49,height:int=512,width:int=512,
                   num_inference_steps:int=30,guidance_scale:float=7.5,seed:int=42,
                    api_kwargs:dict={},model_dir:str="",script_dir:str="",):
    os.chdir(f"{model_dir}/")
    date_time = datetime.now().strftime("%m-%d--%H-%M-%S")
    prompt_file=f"configs/opensora-v1-1/inference/prompt_{date_time}.txt"
    with open(os.path.join(model_dir,prompt_file),"w") as f:
        for item in prompts:
            f.write(repr(item)[1:-1] + '\n')
    
    video_names_file=f"configs/opensora-v1-1/inference/video_names_{date_time}.txt"
    with open(os.path.join(model_dir,video_names_file),"w") as f:
        for video_name in video_names:
            f.write(video_name+"\n")
    
    print("ready to run generating script")

    env = os.environ.copy() 
    env['PYTHONPATH'] = f"{model_dir}"
    env['CUDA_VISIBLE_DEVICES'] = f"{device_id}"
    
    date_time = datetime.now().strftime("%m-%d--%H-%M-%S")
    temp_save_dir=f"{model_dir}/res_v1_1_{date_time}"
    os.makedirs(temp_save_dir,exist_ok=True)
    
    print(f"curr_seed: {seed}")
    subprocess.run(["python", "scripts/inference.py","configs/opensora-v1-1/inference/sample.py",
                "--seed", f"{seed}",
                "--num-frames",f"{num_frames}",
                "--image-size","512","512",
                "--save-dir",f"{raw_video_dir}",
                "--prompt-path",f"{os.path.join(model_dir,prompt_file)}",
                "--layernorm-kernel","False",
                "--num-sampling-steps",f"{num_inference_steps}",
                "--cfg-scale",f"{guidance_scale}",
                "--video-names-path",f"{os.path.join(model_dir,video_names_file)}",
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
    
   
   
def run_opensora_v1_0(prompts:list,raw_video_dir:str,device_id:Union[int, list]=0,video_names:list=[],
                   num_frames:int=16,height:int=512,width:int=512,
                   num_inference_steps:int=30,guidance_scale:float=7.5,seed:int=42,
                    api_kwargs:dict={},model_dir:str="",script_dir:str="",):
    os.chdir(f"{model_dir}/")
    date_time = datetime.now().strftime("%m-%d--%H-%M-%S")
    prompt_file=f"configs/opensora-v1-0/inference/prompt_{date_time}.txt"
    with open(os.path.join(model_dir,prompt_file),"w") as f:
        for item in prompts:
            f.write(repr(item)[1:-1] + '\n')
    
    video_names_file=f"configs/opensora-v1-0/inference/video_names_{date_time}.txt"
    with open(os.path.join(model_dir,video_names_file),"w") as f:
        for video_name in video_names:
            f.write(video_name+"\n")
    
    print("ready to run generating script")

    env = os.environ.copy() 
    env['PYTHONPATH'] = f"{model_dir}"
    env['CUDA_VISIBLE_DEVICES'] = f"{device_id}"
    
    date_time = datetime.now().strftime("%m-%d--%H-%M-%S")
    temp_save_dir=f"{model_dir}/res_v1_0_{date_time}"
    os.makedirs(temp_save_dir,exist_ok=True)
    
    print(f"curr_seed: {seed}")
    subprocess.run([
                "torchrun",
                "--standalone",
                "--nproc_per_node","1",
                "scripts/inference.py","configs/opensora/inference/16x512x512.py",
                "--seed", f"{seed}",
                "--ckpt-path","OpenSora-v1-HQ-16x512x512.pth",
                "--prompt-path",f"{os.path.join(model_dir,prompt_file)}",
                "--layernorm-kernel","False",
                "--save-dir",f"{raw_video_dir}",
                "--num-sampling-steps",f"{num_inference_steps}",
                "--cfg-scale",f"{guidance_scale}",
                "--video-names-path",f"{os.path.join(model_dir,video_names_file)}",
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