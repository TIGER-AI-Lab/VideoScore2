import os
from tqdm import tqdm
from typing import Union
import torch
from diffusers.utils import export_to_video

def run_mochi1_preview_quant(prompts:list,raw_video_dir:str,device_id:Union[int, list]=0,video_names:list=[],
                    num_frames:int=19,height:int=480,width:int=848,
                    num_inference_steps:int=50,guidance_scale:float=4.5,seed:int=42,
                    api_kwargs:dict={},model_dir:str="",script_dir:str="",): 
    torch.manual_seed(seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{device_id}"
    
    from diffusers.quantizers.quantization_config import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
    from diffusers.models.transformers.transformer_mochi import MochiTransformer3DModel
    from diffusers.pipelines.mochi.pipeline_mochi import MochiPipeline
    from transformers import BitsAndBytesConfig as BitsAndBytesConfig, T5EncoderModel
    
    quant_config = BitsAndBytesConfig(load_in_8bit=True)
    text_encoder_8bit = T5EncoderModel.from_pretrained(
        "genmo/mochi-1-preview",subfolder="text_encoder",quantization_config=quant_config,torch_dtype=torch.float16,)

    quant_config = DiffusersBitsAndBytesConfig(load_in_8bit=True)
    transformer_8bit = MochiTransformer3DModel.from_pretrained(
        "genmo/mochi-1-preview", subfolder="transformer", quantization_config=quant_config, torch_dtype=torch.float16,)

    pipe = MochiPipeline.from_pretrained(
        "genmo/mochi-1-preview", text_encoder=text_encoder_8bit, 
        transformer=transformer_8bit, torch_dtype=torch.float16
    ).to("cuda")

    for idx,prompt in tqdm(enumerate(prompts)):
        video_path=os.path.join(raw_video_dir,f"{idx}.mp4")
        if video_names is not None and len(video_names)!=0:
            video_path=os.path.join(raw_video_dir,f"{video_names[idx]}.mp4")
        video_frames = pipe(
            prompt=prompt,
            num_frames=num_frames,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        ).frames[0]
        export_to_video(video_frames,video_path)

 
def run_mochi1_preview_bf(prompts:list,raw_video_dir:str,device_id:Union[int, list]=0,video_names:list=[],
                  num_frames:int=19,height:int=480,width:int=848,
                    num_inference_steps:int=50,guidance_scale:float=4.5,seed:int=42,
                    api_kwargs:dict={},model_dir:str="",script_dir:str="",):
    torch.manual_seed(seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{device_id}"
    
    from diffusers.pipelines.mochi.pipeline_mochi import MochiPipeline
    
    pipe = MochiPipeline.from_pretrained("genmo/mochi-1-preview", variant="bf16", torch_dtype=torch.bfloat16).to("cuda")

    # Enable memory savings
    pipe.enable_model_cpu_offload()
    pipe.enable_vae_tiling()

    for idx,prompt in tqdm(enumerate(prompts)):
        video_path=os.path.join(raw_video_dir,f"{idx}.mp4")
        if video_names is not None and len(video_names)!=0:
            video_path=os.path.join(raw_video_dir,f"{video_names[idx]}.mp4")
        video_frames = pipe(
            prompt=prompt,
            num_frames=num_frames,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        ).frames[0]
        export_to_video(video_frames,video_path)