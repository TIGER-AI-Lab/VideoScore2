import os
import shutil
import subprocess
import yaml
from typing import Union
from datetime import datetime

def run_opensora_plan_v1_3(prompts:list,raw_video_dir:str,device_id:Union[int, list]=0,video_names:list=[],
                   num_frames:int=93,height:int=352,width:int=640,
                   num_inference_steps:int=30,guidance_scale:float=7.5,seed:int=42,
                    api_kwargs:dict={},model_dir:str="",script_dir:str="",):
    os.chdir(f"{model_dir}/")
    prompt_file=f"{model_dir}/examples/prompt_list_v1_3.txt"
    with open(prompt_file,"w") as f:
        for item in prompts:
            f.write(repr(item)[1:-1] + '\n')
    
    video_names_file=f"{model_dir}/examples/video_names.txt"
    with open(video_names_file,"w") as f:
        for video_name in video_names:
            f.write(video_name+"\n")
    
    print("ready to run generating script")
    
    env = os.environ.copy() 
    env['CUDA_VISIBLE_DEVICES'] = f"{device_id}"
    
    subprocess.run([
        "torchrun","--nnodes=1",
        "--nproc_per_node","1",
        "--master_port","29514",
        "-m","opensora.sample.sample",
        "--model_path",f"{model_dir}/ckpt/any93x640x640/diffusion_model",
        "--version","v1_3",
        "--num_frames",f"{num_frames}",
        "--height",f"{height}",
        "--width",f"{width}",
        "--cache_dir","../ckpt",
        "--text_encoder_name_1",f"{model_dir}/ckpt/mt5-xxl",
        "--text_prompt",f"{prompt_file}",
        "--ae","WFVAEModel_D8_4x8x8",
        "--ae_path",f"{model_dir}/ckpt/any93x640x640/vae",
        "--save_img_path",f"{raw_video_dir}",
        "--fps","18",
        "--guidance_scale",f"{guidance_scale}",
        "--num_sampling_steps",f"{num_inference_steps}",
        "--max_sequence_length","512",
        "--sample_method","EulerAncestralDiscrete",
        "--seed",f"{seed}",
        "--num_samples_per_prompt","1",
        "--rescale_betas_zero_snr",
        "--prediction_type","v_prediction",
        "--save_memory",
        "--version","v1_3",
        "--video_names_file",f"{video_names_file}",
        ],env=env)
    
    # video_files=[x for x in sorted(os.listdir(temp_save_dir)) if x.endswith("mp4")]
    # for idx,video_file in enumerate(video_files):
    #     shutil.move(src=os.path.join(temp_save_dir,video_file),dst=os.path.join(raw_video_dir,video_names[idx],".mp4"))
    # os.remove(temp_save_dir)
    
    os.chdir(script_dir)


# def run_opensora_plan_v1_1(prompts:list,raw_video_dir:str,device_id:Union[int, list]=0,video_names:list=[],
#                    num_frames:int=16,height:int=512,width:int=512,
#                    num_inference_steps:int=30,guidance_scale:float=7.5,seed:int=42,
#                     api_kwargs:dict={},model_dir:str="",script_dir:str="",):
#     os.chdir(f"{model_dir}/")
#     prompt_file="examples/prompt_list_0.txt"
#     input_txt_file=f"{model_dir}/{prompt_file}"
#     with open(input_txt_file,"w") as f:
#         for item in prompts:
#             f.write(repr(item)[1:-1] + '\n')
    
#     print("ready to run generating script")
    
#     date_time = datetime.now().strftime("%m-%d--%H-%M-%S")
#     temp_save_dir=f"{model_dir}/examples/res_v1_1_{date_time}"
#     os.makedirs(temp_save_dir,exist_ok=True)
    
#     env = os.environ.copy() 
#     env['CUDA_VISIBLE_DEVICES'] = f"{device_id}"
#     subprocess.run(["python", "opensora/sample/sample.py",
#                     "--seed", f"{seed}",
#                     "--device_id", f"{device_id}",
#                     "--model_path",'opensora/ckpt/v1.1.0/65x512x512',
#                     "--version","65x512x512",
#                     "--num_frames",f"{num_frames}",
#                     "--height", f"{height}",
#                     "--width", f"{width}",
#                     "--ae","CausalVAEModel_D4_4x8x8",
#                     "--ae_path",'opensora/ckpt/v1.1.0/vae',
#                     "--save_img_path", f"{temp_save_dir}",
#                     "--guidance_scale",f"{guidance_scale}",
#                     "--sample_method","EulerAncestralDiscrete",
#                     "--num_sampling_steps",f"{num_inference_steps}",
#                     "--fps","24",
#                     "--text_prompt",f"{prompt_file}",
#                     "--enable_tiling",
#                     "--model_type","dit",
#                     # "--save_memory",
#                     ],env=env)
    
#     # video_files=[x for x in sorted(os.listdir(temp_save_dir)) if x.endswith("mp4")]
#     # for idx,video_file in enumerate(video_files):
#     #     shutil.move(src=os.path.join(temp_save_dir,video_file),dst=os.path.join(raw_video_dir,video_names[idx],".mp4"))
#     # os.remove(temp_save_dir)
    
    
#     os.chdir(script_dir)



# def run_opensora_plan_v1_0(prompts:list,raw_video_dir:str,device_id:Union[int, list]=0,video_names:list=[],
#                    num_frames:int=17,height:int=256,width:int=256,
#                    num_inference_steps:int=30,guidance_scale:float=7.5,seed:int=42,
#                     api_kwargs:dict={},model_dir:str="",script_dir:str="",):
#     os.chdir(f"{model_dir}/")
#     prompt_file="examples/prompt_list_v_1_0.txt"
#     input_txt_file=f"{model_dir}/{prompt_file}"
#     with open(input_txt_file,"w") as f:
#         for item in prompts:
#             f.write(repr(item)[1:-1] + '\n')
    
#     print("ready to run generating script")
    
#     date_time = datetime.now().strftime("%m-%d--%H-%M-%S")
#     temp_save_dir=f"{model_dir}/examples/res_v1_0_{date_time}"
#     os.makedirs(temp_save_dir,exist_ok=True)
    
#     env = os.environ.copy() 
#     env['CUDA_VISIBLE_DEVICES'] = f"{device_id}"
#     subprocess.run(["python", "opensora/sample/sample.py",
#                     "--seed", f"{seed}",
#                     "--device_id", f"{device_id}",
#                     "--model_path",'opensora/ckpt/v1.0.0/17x256x256',
#                     "--version","17x256x256",
#                     "--num_frames",f"{num_frames}",
#                     "--height", f"{height}",
#                     "--width", f"{width}",
#                     "--ae","CausalVAEModel_D4_4x8x8",
#                     "--ae_path",'opensora/ckpt/v1.0.0/vae',
#                     "--save_img_path", f"{temp_save_dir}",
#                     "--guidance_scale",f"{guidance_scale}",
#                     "--sample_method","EulerAncestralDiscrete",
#                     "--num_sampling_steps",f"{num_inference_steps}",
#                     "--fps","24",
#                     "--text_prompt",f"{prompt_file}",
#                     "--enable_tiling",
#                     "--model_type","dit",
#                     # "--save_memory",
#                     ],env=env)

#     # video_files=[x for x in sorted(os.listdir(temp_save_dir)) if x.endswith("mp4")]
#     # for idx,video_file in enumerate(video_files):
#     #     shutil.move(src=os.path.join(temp_save_dir,video_file),dst=os.path.join(raw_video_dir,video_names[idx],".mp4"))
#     # os.remove(temp_save_dir)
    
#     os.chdir(script_dir)