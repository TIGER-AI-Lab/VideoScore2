
import os
import subprocess
from moviepy.editor import VideoFileClip
from typing import Union

def run_hotshot_xl(prompts:list,raw_video_dir:str,device_id:Union[int, list]=0,video_names:list=[],
                  num_frames:int=8,height:int=672,width:int=384,
                    num_inference_steps:int=30,guidance_scale:float=7.0,seed:int=42,
                    api_kwargs:dict={},model_dir:str="",script_dir:str="",): 
    
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{device_id}"
    os.chdir(f"{model_dir}/")
    for idx, prompt in enumerate(prompts):
        video_path=os.path.join(raw_video_dir,f"{idx}.mp4")
        if video_names is not None and len(video_names)!=0:
            video_path=os.path.join(raw_video_dir,f"{video_names[idx]}.mp4")
        print(f"current: {idx} {prompt}")    
        print("ready to run generating script")
        
        subprocess.run(["python", "inference.py","--prompt", f"{prompt}", "--steps", f"{num_inference_steps}","--output", video_path])
        
    os.chdir(script_dir)