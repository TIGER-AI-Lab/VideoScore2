"""From https://github.com/zeno-ml/zeno-build/blob/main/zeno_build/models/providers/openai_utils.py."""
"""Tools to generate from OpenAI prompts."""

import os
import re
import json
from datetime import datetime
from readline import read_history_file
from tqdm import tqdm
import fire
from zeno_build.models import lm_config
import matplotlib.pyplot as plt
import uuid
import pandas as pd
import ast
from utils_gpt_chat import *
import random

NOISE_CHARS="#.*: \n"
ERR_TRIGGER_WORDS=["fps","screen size","16:9","1:1","4:3","3:4","9:16",
                   "4k","8k","seconds","message","font","modern","attach",
                   "say","years old","output","format","high quality"]
WORDS_NUM_MIN=20
WORDS_NUM_MAX=100
root_dir="/data/xuan/videoscore2"


async def koala_prompts():
    
    VIDEO_LEN_MIN=3.0
    VIDEO_LEN_MAX=6.0
    CLARITY_MIN=0.95
    AES_SCORE_MIN=4.0
    VTSS_MIN=3.0
    
    date_time = datetime.now().strftime("%m-%d--%H-%M-%S")
    logger=set_logger(f"./logs/koala_prompts.log")
    
    raw_prompt_path=os.path.join(root_dir,"text_prompts/raw","koala_0_9999.csv")
    res_path=os.path.join(root_dir,"text_prompts","prompt_koala.jsonl")
    os.makedirs(os.path.dirname(res_path),exist_ok=True)
    
    raw_prompt_items = pd.read_csv(raw_prompt_path).to_dict(orient='records')
    print("num of raw prompts: ",len(raw_prompt_items))
    
    # (1) filter by video_length, clarity_score, etc
    
    temp_list=[]
    # caption_lens=[]
    for item in tqdm(raw_prompt_items):
        # caption_lens.append(len(item["caption"].split(" ")))
        time_stamp=ast.literal_eval(item["timestamp"])
        time_format = '%H:%M:%S.%f'
        s_t=datetime.strptime(time_stamp[0], time_format)
        e_t=datetime.strptime(time_stamp[1], time_format)
        duration = (e_t - s_t).total_seconds()
        if duration<VIDEO_LEN_MIN or duration >VIDEO_LEN_MAX:
            continue
        if item["clarity_score"]<CLARITY_MIN \
            or item["aesthetic_score"]<AES_SCORE_MIN \
            or item["video_training_suitability_score"]<VTSS_MIN:
            continue
        temp_list.append(item)
    logger.info(f"num after filtering by metrics: {len(temp_list)}")
    raw_prompt_items=temp_list
    
    # plt.hist(durations, bins=50, edgecolor='black')
    # plt.title('Video Length Dist in Koala')
    # plt.xlabel('Video Length (s)')
    # plt.ylabel('Frequency')
    # plt.savefig("prompt_koala_duration_dist.png")
    
    # plt.hist(caption_lens, bins=50, edgecolor='black')
    # plt.title('Num of Words of Raw Caption Dist in Koala')
    # plt.xlabel('Num of Words (s)')
    # plt.ylabel('Frequency')
    # plt.savefig("prompt_koala_raw_caption_wordnum_dist.png")
    
    # (2) revise by gpt
    BATCH_SIZE=50
    template=CHAT_TEMPLATES[f"koala_compress_caption"]
    
    for i in range(0,len(raw_prompt_items),BATCH_SIZE):
        logger.info(f"\n\n {'#'*50}\n{i//BATCH_SIZE+1}-th batch start!\n\n")
        raw_batch=raw_prompt_items[i:i+BATCH_SIZE]
        revised_prompts=[]
        
        context_list=[]
        for rp in raw_batch:
            user_input=template+"\n### Input prompt: \n"+rp["caption"]
            context=dict(messages=
                [{
                    "content":user_input,
                    "role":"user"
                }])
            context_list.append(context)
        
        res_list = await generate_from_openai_chat_completion(
            full_contexts = context_list,
            model_config = model_config,
            logger=logger,
        )
    
        for idx, res in enumerate(res_list):
            logger.info(f"\n----------------- {idx} raw output -----------------\n {res}")
            
            new_item={"src_id":raw_batch[idx]["videoID"],"timestamp":raw_batch[idx]["timestamp"],"text":""}
            if len(res.split(" ")) < WORDS_NUM_MIN or len(res.split(" ")) > WORDS_NUM_MAX \
                or not str(res).isascii() or re.search(r'\\u[0-9a-fA-F]{4}', res):
                continue
            if '\\"' in res:
                res=res.replace('\\"', "'")
                
            new_item["text"]=res.strip(NOISE_CHARS)
            revised_prompts.append(new_item)
        logger.info(f"\n{'#'*50}\n{i//BATCH_SIZE+1} batch done!\nlen(revised):{len(revised_prompts)}\n\n")
        
        with open(res_path,"a") as f:
            for p in revised_prompts:
                f.write(json.dumps(p)+"\n")
    

async def filter_vidprom_prompts():
    date_time = datetime.now().strftime("%m-%d--%H-%M-%S")
    logger=set_logger(f"./logs/filter_vidprom_prompts_{date_time}.log")
    
    raw_prompt_path=os.path.join(root_dir,"text_prompts/raw","prompt_en_vidprom.jsonl")
    res_path=os.path.join(root_dir,"text_prompts","prompt_vidprom.jsonl")
    os.makedirs(os.path.dirname(res_path),exist_ok=True)
    
    raw_prompt_items=[]
    with open(raw_prompt_path,"r") as f:
        raw_prompt_items=[json.loads(line.strip()) for line in f]
    print("num of raw prompts: ",len(raw_prompt_items))
    
    # coarse filtering by matching
    ## (1) filter by trigger words
    temp_list=[]
    for idx,p in tqdm(enumerate(raw_prompt_items)):
        if any(err_word in str(p["text"]).lower() for err_word in ERR_TRIGGER_WORDS) \
            or not str(p["text"]).isascii() or re.search(r'\\u[0-9a-fA-F]{4}', p["text"]):
            # print(idx,p["text"])
            continue

        temp_list.append(p)
    raw_prompt_items=temp_list
    print("num after filtering by matching: ",len(raw_prompt_items))
    
    ## (2) filter by num of words
    
    # leng_list=[len(x["text"].split(" ")) for x in raw_prompts]
    # leng_list=random.sample(leng_list,1000)
    # plt.hist(leng_list, bins=30, edgecolor='black')
    # plt.title('Word Num Dist in Prompts')
    # plt.xlabel('Num of Words')
    # plt.ylabel('Frequency')
    # plt.savefig("prompt_wordnum_dist.png")
    
    temp_list=[]
    for p in tqdm(raw_prompt_items):
        if len(p["text"].split(" ")) > WORDS_NUM_MAX or len(p["text"].split(" ")) < WORDS_NUM_MIN:
            continue
        temp_list.append(p)
    raw_prompt_items=temp_list
    print("num after filtering by num of words: ",len(raw_prompt_items))    
    
    # (3) finer filtering by gpt
    raw_prompt_items=raw_prompt_items[0:15000]
    # raw_prompt_items=random.sample(raw_prompt_items,100)
    BATCH_SIZE=100
    
    template=CHAT_TEMPLATES[f"vidprom_filter_prompt"]
    
    for i in range(0,len(raw_prompt_items),BATCH_SIZE):
        logger.info(f"\n\n {'#'*50}\n{i//BATCH_SIZE+1}-th batch start!\n\n")
        raw_prompts_batch=raw_prompt_items[i:i+BATCH_SIZE]
        filtered=[]
        
        context_list=[]
        for rp in raw_prompts_batch:
            user_input=template+"\n### Input prompt: \n"+rp["text"]
            context=dict(messages=
                [{
                    "content":user_input,
                    "role":"user"
                }])
            context_list.append(context)
        
        res_list = await generate_from_openai_chat_completion(
            full_contexts = context_list,
            model_config = model_config,
            logger=logger,
        )
    
        for idx, res in enumerate(res_list):
            logger.info(f"\n----------------- {idx} raw output -----------------\n {res}")
            src_id=raw_prompts_batch[idx]["video_name"]
            if "none" in res.lower() or "\n" in res or any(err_word in str(res).lower() for err_word in ERR_TRIGGER_WORDS) \
                 or len(res.split(" ")) < WORDS_NUM_MIN or len(res.split(" ")) > WORDS_NUM_MAX \
                 or not str(res).isascii() or re.search(r'\\u[0-9a-fA-F]{4}', res):
                # logger.info(f"{idx}-th prompt in current batch err: {raw_prompt_items[idx]}")
                continue
            filtered.append({"src_id":src_id,"text":res.strip(NOISE_CHARS)})
        logger.info(f"\n{'#'*50}\n{i//BATCH_SIZE+1} batch done!\nlen(filtered):{len(filtered)}\n\n")

        # with open(res_path,"a") as f:
        #     for p in filtered:
        #         f.write(json.dumps(p)+"\n")

    
async def gen_OCR_text_prompts():
    date_time = datetime.now().strftime("%m-%d--%H-%M-%S")
    logger=set_logger(f"./logs/orc_text_prompt_debug.log")
    res_path=os.path.join(root_dir,"text_prompts","ocr_text_prompt.jsonl")

    context_list=[]
    user_input=CHAT_TEMPLATES["gen_ocr_text_prompt"]
    context=dict(messages=
        [{
            "content":user_input,
            "role":"user"
        }])
    context_list.append(context)
    
    res_list = await generate_from_openai_chat_completion(
        full_contexts = context_list,
        model_config = model_config,
        logger=logger,
    )
    
    ocr_text_prompts=[]
    for idx, res in enumerate(res_list):
        logger.info(f"\n----------------- {idx} raw output -----------------\n {res}")
        temp_batch=res.strip(NOISE_CHARS).split("\n")
        for x in temp_batch:
            if "." in x:
                x=x.split(".")[1]
            if len(x.split(" ")) < WORDS_NUM_MIN-5 \
                            or len(x.split(" ")) > WORDS_NUM_MAX \
                            or not str(x).isascii() \
                            or re.search(r'\\u[0-9a-fA-F]{4}', x) :
                continue
            ocr_text_prompts.append(x.strip(NOISE_CHARS))
    
    uuids=set()
    while len(uuids)<len(ocr_text_prompts):
        uuids.add(uuid.uuid4())
    uuids=list(uuids)
    
    with open(res_path,"a") as f:
        for idx,p in enumerate(ocr_text_prompts):
            f.write(json.dumps({"src_id":str(uuids[idx]),"text":p})+"\n")

    
async def gen_story_prompts():
    date_time = datetime.now().strftime("%m-%d--%H-%M-%S")
    logger=set_logger(f"./logs/gen_seed_story_prompts_debug.log")
    res_path=os.path.join(root_dir,"text_prompts","story_prompt.jsonl")

    context_list=[]
    user_input=CHAT_TEMPLATES["gen_story_prompt"]
    context=dict(messages=
        [{
            "content":user_input,
            "role":"user"
        }])
    context_list.append(context)
    
    res_list = await generate_from_openai_chat_completion(
        full_contexts = context_list,
        model_config = model_config,
        logger=logger,
    )
    
    seed_prompts=[]
    for idx, res in enumerate(res_list):
        logger.info(f"\n----------------- {idx} raw output -----------------\n {res}")
        temp_batch=res.strip(NOISE_CHARS).split("\n")
        temp_batch=[x for x in temp_batch 
                    if not (len(x.split(" ")) < WORDS_NUM_MIN 
                            or len(x.split(" ")) > WORDS_NUM_MAX 
                            or not str(x).isascii() 
                            or re.search(r'\\u[0-9a-fA-F]{4}', x))]
        
        seed_prompts.extend(temp_batch)

    
    uuids=set()
    while len(uuids)<len(seed_prompts):
        uuids.add(uuid.uuid4())
    uuids=list(uuids)
    
    with open(res_path,"a") as f:
        for idx,p in enumerate(seed_prompts):
            f.write(json.dumps({"src_id":str(uuids[idx]),"text":p})+"\n")


async def refine_story_prompts():
    date_time = datetime.now().strftime("%m-%d--%H-%M-%S")
    logger=set_logger(f"./logs/refine_story_prompts.log")
    
    seed_prompt_path=os.path.join(root_dir,"text_prompts","story_prompt.jsonl")
    res_path=os.path.join(root_dir,"text_prompts","story_prompt.jsonl")
    os.makedirs(os.path.dirname(res_path),exist_ok=True)
    
    seed_prompt_items=[]
    with open(seed_prompt_path,"r") as f:
        seed_prompt_items=[json.loads(line.strip()) for line in f]
    print("num of raw prompts: ",len(seed_prompt_items))
    
    context_list=[]
    for seed_prompt in seed_prompt_items:
        user_input=CHAT_TEMPLATES["refine_story_prompt"]+"\n"+seed_prompt["text"]
        context=dict(messages=
            [{
                "content":user_input, 
                "role":"user"
            }])
        context_list.append(context)
    
    res_list = await generate_from_openai_chat_completion(
        full_contexts = context_list,
        model_config = model_config,
        logger=logger,
    )

    refined_prompts=[]
    for idx, res in enumerate(res_list):
        logger.info(f"\n----------------- {idx} raw output -----------------\n {res}")
        if "none" in res.lower() or len(res.split(" ")) < WORDS_NUM_MIN-5 or len(res.split(" ")) > WORDS_NUM_MAX:
            continue
        src_id=seed_prompt_items[idx]["src_id"]
        refined_prompts.append({"src_id":src_id,"text":res.strip(NOISE_CHARS)})
    
    with open(res_path,"w") as f:
        for x in refined_prompts:
            f.write(json.dumps(x)+"\n")
    
 
def add_camera_motion_vidprom():
    CAMERA_MOTIONS=["Zoom in","Zoom out","Pan left","Pan right","Pan up","Pan down",
                    "Tilt up","Tilt down","Tracking shot","Crane up","Crane down"]
    src_path="/data/xuan/videoscore2/text_prompts/prompt_vidprom.jsonl"
    res_path="/data/xuan/videoscore2/text_prompts/prompt_vidprom_camera_motion.jsonl"
    os.makedirs(os.path.dirname(res_path),exist_ok=True)
    prompt_items=[]
    with open(src_path,"r") as f:
        prompt_items=[json.loads(line.strip()) for line in f]
    
    ori_items=prompt_items[:2800]
    for i,x in enumerate(ori_items):
        if "." not in x["text"][-3:]:
            ori_items[i]["text"]+="."
    
    with open(src_path,"w") as f:
        for x in ori_items:
            f.write(json.dumps(x)+"\n")
    
    new_items=prompt_items[-500:]
    for i,x in enumerate(new_items):
        if "." in x["text"][-3:]:
            cam_motion=" "+random.choice(CAMERA_MOTIONS)+"."
        else:
            cam_motion=". "+random.choice(CAMERA_MOTIONS)+"."
        new_items[i]["text"]+=cam_motion
    
    with open(res_path,"w") as f:
        for x in new_items:
            f.write(json.dumps(x)+"\n")
    

def collect_all():
    source_dir="/data/xuan/videoscore2/text_prompts"
    f_names=["prompt_koala.jsonl","prompt_vidprom.jsonl","prompt_vidprom_camera_motion.jsonl","story_prompt.jsonl","ocr_text_prompt.jsonl"]
    src_names=["koala","vidprom","camera_motion","story","ocr_etxt"]
    res_file="all_prompts.jsonl"
    all_items=[]
    data=[]
    for f_idx,f_name in enumerate(f_names):
        with open(os.path.join(source_dir,f_name),"r") as f:
            data=[json.loads(line) for line in f] 
        for i,x in enumerate(data):
            new_item={"video_id":"","text":x["text"],"src":src_names[f_idx],"src_id":x["src_id"],}
            all_items.append(new_item)
            
    random.shuffle(all_items)
    
    for i,x in enumerate(all_items):
        x["video_id"]=f"{i:06d}"
    
    with open(os.path.join(source_dir,res_file),"w") as f2:
        for p in all_items:
            f2.write(json.dumps(p)+"\n")
    

async def translate_all():
    date_time = datetime.now().strftime("%m-%d--%H-%M-%S")
    logger=set_logger(f"./logs/orc_text_prompt_debug.log")
    
    src_path="/data/xuan/videoscore2/text_prompts/all_prompts.jsonl"
    res_path="/data/xuan/videoscore2/text_prompts/all_prompts_en_cn.jsonl"
    with open(src_path,"r") as f:
        all_items=[json.loads(line) for line in f]
    
    BATCH_SIZE=100
    template=CHAT_TEMPLATES["translate"]
    for i in range(0,len(all_items),BATCH_SIZE):
        logger.info(f"\n\n {'#'*50}\n{i//BATCH_SIZE+1}-th batch start!\n\n")
        items_batch=all_items[i:i+BATCH_SIZE]
        done=[]
        
        context_list=[]
        for rp in items_batch:
            user_input=template+"\n### Input prompt: \n"+rp["text"]
            context=dict(messages=
                [{
                    "content":user_input,
                    "role":"user"
                }])
            context_list.append(context)
        
        res_list = await generate_from_openai_chat_completion(
            full_contexts = context_list,
            model_config = model_config,
            logger=logger,
        )
    
        for idx, res in enumerate(res_list):
            logger.info(f"\n----------------- {idx} raw output -----------------\n {res}")
            new_item=items_batch[idx]
            new_item["text_cn"]=res.strip(NOISE_CHARS)
            done.append(new_item)
        logger.info(f"\n{'#'*50}\n{i//BATCH_SIZE+1} batch done!\nlen(done):{len(done)}\n\n")
    
        with open(res_path,"a") as f:
            for x in done:
                f.write(json.dumps(x)+"\n")
    

if __name__ == "__main__":
    CHAT_TEMPLATES=json.load(open("../const/gpt_chat_template.json","r",encoding='utf-8'))
    
    API_KEYS=json.load(open("../const/api_key.json","r",encoding='utf-8'))
    os.environ["OPENAI_API_KEY"]=API_KEYS["OpenAI_API_KEYd1"]
    # os.environ["OPENAI_ORG"]=API_KEYS["OpenAI_ORG_ID"]
    os.environ["OPENAI_BASE_URL"]=API_KEYS["DeepBricks_BASE_URL"]
    
    # model_name="gpt-4o-2024-08-06"
    model_name = "gpt-4o-mini"
    model_config = lm_config.LMConfig(provider="openai_chat", model=model_name)
    
    # fire.Fire(koala_prompts)
    # fire.Fire(filter_vidprom_prompts)
    # fire.Fire(add_camera_motion_vidprom)
    # fire.Fire(gen_OCR_text_prompts)
    # fire.Fire(gen_story_prompts)
    # fire.Fire(refine_story_prompts)
    # fire.Fire(collect_all)
    fire.Fire(translate_all)

    