import os
from tqdm import tqdm
tqdm.disable = True

from typing import Union
import torch
from diffusers.utils import export_to_video

def run_modelscope(prompts:list,raw_video_dir:str,device_id:Union[int, list]=0,video_names:list=[],
                  num_frames:int=24,height:int=256,width:int=256,
                    num_inference_steps:int=50,guidance_scale:float=9.0,seed:int=42,
                    api_kwargs:dict={},model_dir:str="",script_dir:str="",): 
    
    torch.manual_seed(seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{device_id}"
    
    # from diffusers import DiffusionPipeline
    from diffusers.pipelines.pipeline_utils import DiffusionPipeline
    
    pipe = DiffusionPipeline.from_pretrained("damo-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float16, variant="fp16")
    pipe.enable_model_cpu_offload()

    # memory optimization
    pipe.enable_vae_slicing()
    
    for idx,prompt in tqdm(enumerate(prompts)):
        video_path=os.path.join(raw_video_dir,f"{idx}.mp4")
        if video_names is not None and len(video_names)!=0:
            video_path=os.path.join(raw_video_dir,f"{video_names[idx]}.mp4")
        video_frames = pipe(
            prompt=prompt,
            num_frames=num_frames,
            # height=height,
            # width=width,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
        ).frames[0]
        export_to_video(video_frames,video_path)
        
        if idx % 10 == 0:
            print(f"modelscope: {idx}")
    print("done")