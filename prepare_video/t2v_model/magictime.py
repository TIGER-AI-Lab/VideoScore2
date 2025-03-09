import os
import shutil
import yaml
import subprocess
from typing import Union
from datetime import datetime

    
def run_magictime(prompts:list,raw_video_dir:str,device_id:Union[int, list]=0,video_names:list=[],
                   num_frames:int=16,height:int=512,width:int=512,
                   num_inference_steps:int=25,guidance_scale:float=8.5,seed:int=42,
                    api_kwargs:dict={},model_dir:str="",script_dir:str="",):
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{device_id}"
    date_time = datetime.now().strftime("%m-%d--%H-%M-%S")
    
    config_file=f"/sample_configs/RealisticVision.yaml"
    curr_config_file=f"sample_configs/RealisticVision_{date_time}.yaml"
    os.chdir(f"{model_dir}/")

            
    video_names_file=f"sample_configs/video_names.txt"
    with open(os.path.join(model_dir,video_names_file),"w") as f:
        for video_name in video_names:
            f.write(video_name+"\n")
    
    config = yaml.safe_load(open(f"{model_dir}/{config_file}", 'r'))
     
    config[0]['steps'] = num_inference_steps
    config[0]['guidance_scale'] = guidance_scale
    config[0]['L'] = num_frames
    config[0]['H'] = height
    config[0]['W'] = width
    config[0]['seed'] = seed
    config[0]['prompt'] = prompts
    config[0]['video_names_file']=f"{os.path.join(model_dir,video_names_file)}",
    print("curr_seed", seed)
        
    with open(curr_config_file, 'w') as f:
        yaml.dump(config, f)    
        
    print("ready to run generating script")
    subprocess.run(["python", "inference_magictime.py",
                    "--save_path", f"{raw_video_dir}",
                    "--config", f"{model_dir}/{curr_config_file}"])

    os.chdir(script_dir)
    # shutil.move(backup_file, config_file)
