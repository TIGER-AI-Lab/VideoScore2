import os
from tqdm import tqdm
from typing import Union
import torch
from diffusers.utils import export_to_video

def run_hunyuan_video(prompts:list,raw_video_dir:str,device_id:Union[int, list]=0,video_names:list=[],
                    num_frames:int=129,height:int=720,width:int=1280,
                    num_inference_steps:int=50,guidance_scale:float=6.0,seed:int=42,
                    api_kwargs:dict={},model_dir:str="",script_dir:str="",): 
    torch.manual_seed(seed) 
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{device_id}"
    
    from diffusers.quantizers.quantization_config import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
    from diffusers.models.transformers.transformer_hunyuan_video import HunyuanVideoTransformer3DModel
    from diffusers.pipelines.hunyuan_video.pipeline_hunyuan_video import HunyuanVideoPipeline

    quant_config = DiffusersBitsAndBytesConfig(load_in_8bit=True)
    transformer_8bit = HunyuanVideoTransformer3DModel.from_pretrained(
        "hunyuanvideo-community/HunyuanVideo",subfolder="transformer",quantization_config=quant_config,torch_dtype=torch.bfloat16,
    )
    

    pipe = HunyuanVideoPipeline.from_pretrained(
        "hunyuanvideo-community/HunyuanVideo",transformer=transformer_8bit,torch_dtype=torch.float16,
    ).to("cuda")
    
    for idx,prompt in tqdm(enumerate(prompts)):
        video_path=os.path.join(raw_video_dir,f"{idx}.mp4")
        if video_names is not None and len(video_names)!=0:
            video_path=os.path.join(raw_video_dir,f"{video_names[idx]}.mp4")
        video_frames = pipe(
            prompt=prompt,
            num_frames=num_frames,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        ).frames[0]
        export_to_video(video_frames,video_path)

