from vchitect2 import run_vchitect2
from magictime import run_magictime
from videolavit import run_videolavit
from opensora import *
from opensora_plan import *
from skyreels_v1 import *
from stepvideo_t2v import *
from wanx21 import *
from hunyuanvideo import *
from moviepy.editor import VideoFileClip
import torch
import os

def video_info(video_path):
    clip = VideoFileClip(video_path)
    print(clip.w)
    print(clip.h)
    print(clip.fps)
    print(clip.duration)
    print(clip.audio)

def run():
    rep_token="r8_CaA3gW8F5C6D5C0uMvSbpGNzO50QQCn1TwBOy"
    prompts=["A boy and a dog are playing soccer on a grassy field, looking very happy. The boy is wearing a red jersey, and the dog is a golden retriever. Trees and flowers surround them, and the sky is blue with a few white clouds.",
             ]
    raw_video_dir="/data/xuan/VideoScore2/prepare_video/t2v_model/examples"
    seed=42
    t2v_model_dir="/data/xuan/videoscore2/t2v_model"
    script_dir=os.path.dirname(os.path.abspath(__file__))
    
    
    model_dir=""
    guidance_scale=6.0
    num_inference_steps=50
    
    num_frames=24
    h=1024
    w=1024
    
    # # device cuda and cpu
    # video_names=["latte_quant"]
    # run_latte_quant(prompts=prompts,raw_video_dir=raw_video_dir,
    #                 seed=seed,device_id=device_id,
    #                 model_dir=model_dir,script_dir=script_dir,
    #                 video_names=video_names,
    #                 )
    
    # # device_id seems to be wrong
    # video_names=["mochi1_preview_bf"]
    # run_mochi1_preview_bf(prompts=prompts,raw_video_dir=raw_video_dir,
    #                 seed=seed,device_id=device_id,
    #                 model_dir=model_dir,script_dir=script_dir,
    #                 video_names=video_names,
    #                 )
    
    
    # api_kwargs={
    #     "token":rep_token,
    #     "duration":4,
    #     "aspect_ratio":"4:3"
    # }
    # video_names=["haiper"]
    # run_haiper(prompts=prompts,raw_video_dir=raw_video_dir,
    #                 model_dir=model_dir,script_dir=script_dir,
    #                 video_names=video_names,api_kwargs=api_kwargs,
    #                 )
    
    # wget https://huggingface.co/Vchitect/LaVie/resolve/main/lavie_base.pt
    
    # =========== to be tested ==============
    # OOM
    # video_names=["videolavit"]
    # device_id=6
    # run_videolavit(prompts=prompts,raw_video_dir=raw_video_dir,
    #                 video_names=video_names,device_id=device_id,
    #                 model_dir=f"{t2v_model_dir}/LaVIT/VideoLaVIT",script_dir=script_dir,
    #                 )
    # print(f"run_{video_names[0]} done")
    
    
    # video_names=["vchitect2"]
    # device_id=7
    # run_vchitect2(prompts=prompts,raw_video_dir=raw_video_dir,
    #                 video_names=video_names,device_id=device_id,
    #                 model_dir=f"{t2v_model_dir}/Vchitect-2.0",script_dir=script_dir,
    #                 )
    # print(f"run_{video_names[0]} done")
    
    
    # video_names=["opensora_v1_2"]
    # device_id=7
    # run_opensora_v1_2(prompts=prompts,raw_video_dir=raw_video_dir,
    #                 video_names=video_names,device_id=device_id,
    #                 model_dir=f"{t2v_model_dir}/Open-Sora",script_dir=script_dir,
    #                 )
    # print(f"run_{video_names[0]} done")
    

    # video_names=["stepvideo_t2v_low_vram",]
    # device_id=6
    # run_stepvideo_t2v_low_vram(prompts=prompts,raw_video_dir=raw_video_dir,
    #                 video_names=video_names,device_id=device_id,
    #                 model_dir=f"{t2v_model_dir}/DiffSynth-Studio",script_dir=script_dir,
    #                 )
    # print(f"run_{video_names[0]} done")         

    
    # video_names=["wanx21_1_3b",] 
    # device_id=6
    # run_wanx21_1_3b(prompts=prompts,raw_video_dir=raw_video_dir,
    #                 video_names=video_names,device_id=device_id,
    #                 model_dir=f"{t2v_model_dir}/DiffSynth-Studio",script_dir=script_dir,
    #                 )
    # print(f"run_{video_names[0]} done")
    
    
    video_names=["hunyuanvideo_24g",] 
    device_id=5
    run_hunyuanvideo_24g(prompts=prompts,raw_video_dir=raw_video_dir,
                    video_names=video_names,device_id=device_id,
                    model_dir=f"{t2v_model_dir}/DiffSynth-Studio",script_dir=script_dir,
                    )
    print(f"run_{video_names[0]} done")
    
    
    # video_names=["skyreels_v1"]
    # device_id=5
    # run_skyreels_v1(prompts=prompts,raw_video_dir=raw_video_dir,
    #                 video_names=video_names,device_id=device_id,
    #                 model_dir=f"{t2v_model_dir}/SkyReels-V1",script_dir=script_dir,
    #                 )
    # print(f"run_{video_names[0]} done")
    
    # video_info(os.path.join(raw_video_dir,f"{video_names[0]}.mp4"))
    
    # /data/xuan/videoscore2/t2v_model/Open-Sora-Plan/ckpt/any93x640x640/diffusion
    # 



if __name__ == "__main__":
    run()
    
    
    # url="https://replicate.delivery/xezq/ppZsjat9gjJXGRcJ0oOkBohRzOkiVcdXrKLdWYo17EQgVUEF/tmpn0u3oc74.mp4"
    # video_path="/data/xuan/video_eval/leader_bd/videos/kling/w_0007_0.mp4"
    # response = requests.get(url)
    # if response.status_code == 200:
    #     with open(video_path, "wb") as file:
    #         file.write(response.content)
    # video_info(video_path)
    
    