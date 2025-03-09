# from t2v_model.anidiff import run_anidiff
# from t2v_model.latte import run_latte
# from t2v_model.mochi1_preview import run_mochi1_preview_quant
# from t2v_model.modelscope import run_modelscope
from t2v_model.cogvideox import run_cogvideox_2b,run_cogvideox_5b,run_cogvideox15_5b
# from t2v_model.lavie import run_lavie_base
# from t2v_model.ltx_video import run_ltx_video
# from t2v_model.zeroscope import run_zeroscope
# from t2v_model.text2video_zero import run_text2video_zero
# from t2v_model.hotshot_xl import run_hotshot_xl
# from t2v_model.videocrafter2 import run_videocrafter2
from t2v_model.vchitect2 import run_vchitect2
# from t2v_model.magictime import run_magictime
# from t2v_model.opensora import run_opensora_v1_3
# from t2v_model.opensora_plan import run_opensora_plan_v1_3
import json
import os
import fire 

model_code_mapping={
                    "anidiff":"a",
                   "latte":"b",
                   "mochi1_preview":"c",
                   "modelscope":"d",
                   "cogvideox_2b":"e",
                   "cogvideox_5b":"f",
                   "cogvideox15_5b":"g",
                   "lavie_base":"h",
                   "ltx_video":"i",
                   "zeroscope":"j",
                   "text2video_zero":"k",
                   "hotshot_xl":"m",
                   "videocrafter2":"n",
                   "vchitect2":"p",
                   "magictime":"q",
                   "kling":"r",
                   "sora":"s",
                   "pika_v2_2":"t",
                   "opensora_plan_v1_3":"u",
                   }

model_pipe_mapping={
                #     "anidiff":run_anidiff,
                #     "latte":run_latte,
                #     "mochi1_preview":run_mochi1_preview_quant,
                #     "modelscope":run_modelscope,
                    "cogvideox_2b":run_cogvideox_2b,
                #    "cogvideox_5b":run_cogvideox_5b,
                #    "cogvideox15_5b":run_cogvideox15_5b,
                #    "lavie_base":run_lavie_base,
                #    "ltx_video":run_ltx_video,
                #    "zeroscope":run_zeroscope,
                #    "text2video_zero":run_text2video_zero,
                #    "hotshot_xl":run_hotshot_xl,
                #    "videocrafter2":run_videocrafter2,
                   "vchitect2":run_vchitect2,
                #    "magictime":run_magictime,
                #    "opensora_plan_v1_3":run_opensora_plan_v1_3,
                    }

rep_token="r8_CaA3gW8F5C6D5C0uMvSbpGNzO50QQCn1TwBOy"

def gen_video(t2v_model,device_id,start_idx,end_idx):
    root_dir="/data/xuan/videoscore2"
    prompt_path=os.path.join(root_dir,"text_prompts","all_prompts.jsonl")
    prompts=[]
    with open(prompt_path,"r") as f:
        for line in f:
            d=json.loads(line.strip())
            prompts.append(d["text"])
    
    # start_idx=2000
    # end_idx=start_idx + 999
    prompts=prompts[start_idx:end_idx+1]
    
    seed=42
    model_code=model_code_mapping[t2v_model]
    video_names=[f"{i:06d}_{model_code}" for i in range(start_idx,end_idx + 1)]
        
    pipe_function=model_pipe_mapping[t2v_model]
    
    raw_video_dir=os.path.join(root_dir,"videos",t2v_model)
    os.makedirs(raw_video_dir,exist_ok=True)
    
    # hf_pipe
    hf_pipe_list=["anidiff","latte","ltx_video","mochi1_preview","modelscope","zeroscope","text2video_zero","cogvideox_2b","cogvideox_5b","cogvideox15_5b",]
    if t2v_model in hf_pipe_list:
        pipe_function(prompts=prompts,raw_video_dir=raw_video_dir,
                    video_names=video_names,device_id=device_id,seed=seed
                    )
    
    # replicate api
    rep_api_list=["pyramid_flow","kling_v16_standard",]
    
    if t2v_model in rep_api_list:
        if t2v_model=="lavie_base":
            api_kwargs={"token":rep_token}
        pipe_function(prompts=prompts,raw_video_dir=raw_video_dir,
                    video_names=video_names,api_kwargs=api_kwargs,
                    )
        
    # source code
    src_code_list=["hotshot_xl","lavie_base","videocrafter2","magictime","vchitect2","opensora_plan_v1_3","opensora_v1_3","videolavit",]
    if t2v_model in src_code_list:
        script_dir=os.path.dirname(os.path.abspath(__file__))
        model_dir=""
        if t2v_model=="hotshot_xl":
            model_dir=os.path.join(root_dir,"t2v_model","Hotshot-XL")
        if t2v_model=="lavie_base":
            model_dir=os.path.join(root_dir,"t2v_model","LaVie")
        if t2v_model=="videocrafter2":
            model_dir=os.path.join(root_dir,"t2v_model","VideoCrafter")
        if t2v_model=="vchitect2":
            model_dir=os.path.join(root_dir,"t2v_model","Vchitect-2.0")
        if t2v_model=="videolavit":
            model_dir=os.path.join(root_dir,"t2v_model","LaVIT","VideoLaVIT")
        if t2v_model=="magictime":
            model_dir=os.path.join(root_dir,"t2v_model","MagicTime")
        if t2v_model=="opensora_plan_v1_3":
            model_dir=os.path.join(root_dir,"t2v_model","Open-Sora-Plan")
        pipe_function(prompts=prompts,raw_video_dir=raw_video_dir,
                    video_names=video_names,device_id=device_id,seed=seed,
                    model_dir=model_dir,script_dir=script_dir,
                    )
    
    
if __name__ == "__main__":
    
    fire.Fire(gen_video)
    
    # python gen_video.py --t2v_model vchitect2 --device_id 3 --start_idx 2500 --end_idx 2999
    
    # cd /home/brantley/workdir/VideoScore2/prepare_video
    # yt-dlp -f "bestvideo[ext=mp4]+bestaudio[ext=mp4]" -o "my_video.mp4" --download-sections "*0:05:47.080-0:06:04.297" https://www.youtube.com/watch?v=D03BQb0sEqw