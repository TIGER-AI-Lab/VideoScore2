import os
from tqdm import tqdm
from typing import Union
import torch
from diffusers.utils import export_to_video

def run_NOVA(prompts:list,raw_video_dir:str,device_id:Union[int, list]=0,video_names:list=[],
                num_frames:int=16,height:int=512,width:int=512,
                    num_inference_steps:int=128,guidance_scale:float=7.5,seed:int=42,
                    api_kwargs:dict={},model_dir:str="",script_dir:str="",):   
    
    torch.manual_seed(seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{device_id}"
    
    from diffnext.pipelines import NOVAPipeline

    model_id = "BAAI/nova-d48w1024-osp480"
    model_args = {"torch_dtype": torch.float16, "trust_remote_code": True}
    pipe = NOVAPipeline.from_pretrained(model_id, **model_args)
    pipe = pipe.to("cuda")
    
    for idx,prompt in tqdm(enumerate(prompts)):
        video_path=os.path.join(raw_video_dir,f"{idx}.mp4")
        if video_names is not None and len(video_names)!=0:
            video_path=os.path.join(raw_video_dir,f"{video_names[idx]}.mp4")
        video_frames = pipe(
            prompt=prompt,
            # num_frames=num_frames,
            # height=height,
            # width=width,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            num_diffusioin_steps=100,
            max_latent_length=9,
        ).frames[0]
        export_to_video(video_frames,video_path,fps=12)
        
  


