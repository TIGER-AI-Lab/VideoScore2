
import os
from tqdm import tqdm
import requests
import yaml
from datetime import datetime
import shutil
import subprocess
from typing import Union

def run_lavie_base(prompts:list,raw_video_dir:str,device_id:Union[int, list]=0,video_names:list=[],
                   num_frames:int=129,height:int=720,width:int=1280,
                   num_inference_steps:int=50,guidance_scale:float=7.5,seed:int=42,
                    api_kwargs:dict={},model_dir:str="",script_dir:str="",):
    
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{device_id}"
    date_time = datetime.now().strftime("%m-%d--%H-%M-%S")
    config_file=f"{model_dir}/base/configs/sample.yaml"
    curr_config_file=f"{model_dir}/base/configs/sample_{date_time}.yaml"
    backup_file=f"{model_dir}/base/configs/backup.yaml"

    config = yaml.safe_load(open(config_file, 'r'))
    config['text_prompt'] = prompts
    config['output_folder'] = f"{raw_video_dir}/"
    config['num_sampling_steps'] = num_inference_steps
    config['guidance_scale'] = guidance_scale
    
    config['video_names']=video_names
    
    with open(curr_config_file, 'w') as f:
        yaml.dump(config, f)

    shutil.copyfile(config_file,backup_file)
    os.chdir(f"{model_dir}/base/")
    print("ready to run generating script")
    subprocess.run(["python", "pipelines/sample.py","--config", f"{model_dir}/base/{curr_config_file}"])
    print("current env: ",os.environ.get('CONDA_DEFAULT_ENV'))
    os.chdir(script_dir)



# def run_lavie_base_replicate(prompts:list,raw_video_dir:str,video_names:list=[],
#                    num_inference_steps:int=50,guidance_scale:float=7.5,seed:int=42,
#                     api_kwargs:dict={},):
#     import replicate
#     err_log=f"./api_err_report/lavie_base.txt"
    
#     if "token" not in api_kwargs.keys():
#         raise ValueError("arg error: api_kwargs-token")
    
#     if "quality" not in api_kwargs.keys():
#         api_kwargs["quality"] = 9
#     if api_kwargs["quality"] not in range(3,10+1):
#         raise ValueError("arg error: api_kwargs-quality")
    
#     os.environ["REPLICATE_API_TOKEN"]=api_kwargs["token"]
    
#     for idx,prompt in tqdm(enumerate(prompts)):
#         video_path=os.path.join(raw_video_dir,f"{idx}.mp4")
#         if video_names is not None and len(video_names)!=0:
#             video_path=os.path.join(raw_video_dir,f"{video_names[idx]}.mp4")
#         input={
        
#             "prompt": prompt,
#             "width": 512,
#             "height": 320,
#             "guidance_scale": guidance_scale,
#             "num_inference_steps": num_inference_steps,
#             "seed":seed,
#             "quality": 9,
#             "video_fps": 8,
#             "sample_method": "ddpm",
#             "interpolation": False,
#             "super_resolution": False,
            
#         }
#         try:
#             output = replicate.run(
#                 "cjwbw/lavie:0bca850c4928b6c30052541fa002f24cbb4b677259c461dd041d271ba9d3c517",
#                 input=input
#             )
#             response = requests.get(output)
#             if response.status_code == 200:
#                 with open(video_path, "wb") as file:
#                     file.write(response.content)
#             else:
#                 raise ValueError("download error, status code:", response.status_code)
#         except Exception as e:
#             date_time = datetime.now().strftime("%m-%d--%H-%M-%S")
#             if "download error" in str(e):
#                 err_text=f"{date_time}, Download error: {idx}-th prompt"
#             else:
#                 err_text=f"{date_time}, API calling error: {idx}-th prompt"
                
#             with open(err_log, "a") as file:
#                 file.write(err_text + "\n")
            
        
        


