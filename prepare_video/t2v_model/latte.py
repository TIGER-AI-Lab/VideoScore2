import os
from tqdm import tqdm
from typing import Union
import torch
from diffusers.utils import export_to_video

def run_latte(prompts:list,raw_video_dir:str,device_id:Union[int, list]=0,video_names:list=[],
                  num_frames:int=16,height:int=512,width:int=512,
                    num_inference_steps:int=100,guidance_scale:float=7.0,seed:int=42,
                    api_kwargs:dict={},model_dir:str="",script_dir:str="",): 
    torch.manual_seed(seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{device_id}"

    from diffusers.pipelines.latte.pipeline_latte import LattePipeline
    from transformers import BitsAndBytesConfig as BitsAndBytesConfig

    pipe = LattePipeline.from_pretrained(
        "maxin-cn/Latte-1", torch_dtype=torch.float16
    ).to("cuda")
    pipe.transformer.to(memory_format=torch.channels_last)
    pipe.vae.to(memory_format=torch.channels_last)

    # pipe.transformer = torch.compile(pipe.transformer)
    # pipe.vae.decode = torch.compile(pipe.vae.decode)
    
    for idx,prompt in tqdm(enumerate(prompts)):
        video_path=os.path.join(raw_video_dir,f"{idx}.mp4")
        if video_names is not None and len(video_names)!=0:
            video_path=os.path.join(raw_video_dir,f"{video_names[idx]}.mp4")
        video_frames = pipe(
            prompt=prompt,
            video_length=num_frames,
            # height=height,
            # width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        ).frames[0]
        export_to_video(video_frames,video_path)


def run_latte_quant(prompts:list,raw_video_dir:str,device_id:Union[int, list]=0,video_names:list=[],
                  num_frames:int=16,height:int=512,width:int=512,
                    num_inference_steps:int=100,guidance_scale:float=7.0,seed:int=42,
                    api_kwargs:dict={},model_dir:str="",script_dir:str="",): 
    torch.manual_seed(seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{device_id}"
    
    from diffusers.quantizers.quantization_config import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
    from diffusers.models.transformers.latte_transformer_3d import LatteTransformer3DModel
    from diffusers.pipelines.latte.pipeline_latte import LattePipeline
    from transformers import BitsAndBytesConfig as BitsAndBytesConfig, T5EncoderModel

    quant_config = BitsAndBytesConfig(load_in_8bit=True)
    text_encoder_8bit = T5EncoderModel.from_pretrained(
        "maxin-cn/Latte-1", subfolder="text_encoder", quantization_config=quant_config,torch_dtype=torch.float16,
    )

    quant_config = DiffusersBitsAndBytesConfig(load_in_8bit=True)
    transformer_8bit = LatteTransformer3DModel.from_pretrained(
        "maxin-cn/Latte-1", subfolder="transformer", quantization_config=quant_config, torch_dtype=torch.float16,
    )

    pipe = LattePipeline.from_pretrained(
        "maxin-cn/Latte-1", text_encoder=text_encoder_8bit, transformer=transformer_8bit, torch_dtype=torch.float16,
    ).to("cuda")
    
    for idx,prompt in tqdm(enumerate(prompts)):
        video_path=os.path.join(raw_video_dir,f"{idx}.mp4")
        if video_names is not None and len(video_names)!=0:
            video_path=os.path.join(raw_video_dir,f"{video_names[idx]}.mp4")
        video_frames = pipe(
            prompt=prompt,
            video_length=num_frames,
            # height=height,
            # width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        ).frames[0]
        export_to_video(video_frames,video_path)