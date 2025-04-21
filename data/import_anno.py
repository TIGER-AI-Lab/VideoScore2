import json
import os
import shutil
import sys
import time
import re
import asyncio
from zeno_build.models import lm_config
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
    print(sys.path)
from prepare_video.prompts.utils_gpt_chat import *



"""item format:
{
    "_id": "",
    "batchId": "",
    "info": {
        "data": [
            {
                "content": "PROMPT",
                "type": "TITLE"
            },
            {
                "content": "English Prompt: The sun rises over a serene city landscape, transitioning to bustling streets as fans in vibrant football jerseys converge towards iconic Premier League stadiums. The energy is palpable, with the excitement building for a day packed with football action. \n翻译为中文的Prompt:阳光照耀着宁静的城市风景，逐渐转向熙熙攘攘的街道，身穿鲜艳足球球衣的球迷们聚集向标志性的英超体育场。氛围令人振奋，兴奋感在为充满足球赛事的一天而不断升温。",
                "type": "TEXT"
            },
            {
                "content": "https://molar-public.oss-cn-hangzhou.aliyuncs.com/VideoScore/1000_1499/ltx_video_091/001023_i.mp4",
                "type": "VIDEO"
            },
            {
                "content": "src_id: f731fee5-c9f6-57da-a98f-03be1b16df73  src: vidprom",
                "type": "TEXT"
            }
        ]
    },
    "labels": [
        {
            "_id": "",
            "data": {
                "id": 1,
                "hash": "1_视觉质量评分",
                "label": "1_视觉质量评分",
                "value": 3, (or "value": "3-Medium",)
                "drawType": "QUESTION",
                "count": 1
            }
        },
        {
            "_id": "",
            "data": {
                "id": 2,
                "hash": "1_视觉质量描述",
                "label": "1_视觉质量描述",
                "value": "画质比较模糊，看不清城市风景，街道和标志性的体育场",
                "drawType": "QUESTION",
                "count": 1
            }
        },
        ......
    ]
},
"""

data_dir="/data/xuan/videoscore2/data"
all_data_path=f"{data_dir}/all_anno.json"
COMMENT_REFINE_TEMPLATE=""
shared_comments=json.load(open("./const/shared_comments.json","r"))
MIN_SCORE=1
MAX_SCORE=5

def convert_anno(raw_anno_file):
    with open(raw_anno_file,"r",encoding="utf-8") as f:
        raw_annos=json.load(f)
    data=[]
    visual_scores=[]
    visual_comments=[]
    t2v_scores=[]
    t2v_comments=[]
    physical_scores=[]
    physical_comments=[]
    for anno in raw_annos:
        url=anno["info"]["data"][2]["content"]
        video_name=url.split("/")[-1].split(".")[0]
        prompt_en=anno["info"]["data"][1]["content"].split("English Prompt")[1].split("\n")[0].strip(". :\n")
        prompt_cn=anno["info"]["data"][1]["content"].split("翻译为中文的Prompt")[1].split("\n")[0].strip(". :\n")
        try:
            visual_score=re.search(r'\d+', str(anno["labels"][0]["data"]["value"])).group()
        except:
            Warning(f"visual score not found for {video_name}")
            continue
        try:
            t2v_score=re.search(r'\d+', str(anno["labels"][2]["data"]["value"])).group()
        except:
            Warning(f"t2v score not found for {video_name}")
            continue
        try:
            physical_score=re.search(r'\d+', str(anno["labels"][4]["data"]["value"])).group()
        except:
            Warning(f"physical score not found for {video_name}")
            continue
        
        visual_comment=anno["labels"][1]["data"]["value"]
        t2v_comment=anno["labels"][3]["data"]["value"]
        physical_comment=anno["labels"][5]["data"]["value"]
        if visual_score==MIN_SCORE or visual_score==MAX_SCORE:
            visual_comment="NA"
        if t2v_score==MAX_SCORE:
            t2v_comment="NA"
        if physical_score==MAX_SCORE:
            physical_comment="NA"     
               
        visual_scores.append(visual_score)
        visual_comments.append(visual_comment)
        t2v_scores.append(t2v_score)
        t2v_comments.append(t2v_comment)
        physical_scores.append(physical_score)
        physical_comments.append(physical_comment)
     
        data_item={
            "video_name":video_name,
            "prompt":prompt_en,
            "visual":{
                "score":visual_score,
                "comment":None,
            },
            "t2v_align":{
                "score":t2v_score,
                "comment":None,
            },
            "physical":{
                "score":physical_score,
                "comment":None,
            }
        }
        data.append(data_item)
    
    visual_comments_refined=asyncio.run(refine_comment_gpt(visual_comments))
    t2v_comments_refined=asyncio.run(refine_comment_gpt(t2v_comments))
    physical_comments_refined=asyncio.run(refine_comment_gpt(physical_comments))
    
    for idx in range(len(visual_scores)):     
        data[idx]['visual']["comment"]=visual_comments_refined[idx]
        data[idx]['t2v_align']["comment"]=t2v_comments_refined[idx]
        data[idx]['physical']["comment"]=physical_comments_refined[idx]
        if visual_scores[idx]==MIN_SCORE:
            data[idx]['visual']["comment"]=shared_comments["visual_1"]
        if visual_scores[idx]==MAX_SCORE:
            data[idx]['visual']["comment"]=shared_comments["visual_5"]
        if t2v_scores[idx]==MIN_SCORE:
            data[idx]['t2v_align']["comment"]=shared_comments["t2v_1"]
        if physical_scores[idx]==MIN_SCORE:
            data[idx]['physical']["comment"]=shared_comments["physical_1"]
        
        
    # with open(save_path,"a",encoding="utf-8") as f:
    #     json.dump(data,f,indent=4,ensure_ascii=False)
    
    
    
async def refine_comment_gpt(comments,):
    model_name="gpt-4o-2024-08-06"
    # model_name = "gpt-4o-mini"
    model_config = lm_config.LMConfig(provider="openai_chat", model=model_name)
    
    date_time=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    os.makedirs(f"./logs",exist_ok=True)
    logger=set_logger(f"./logs/import_anno_{date_time}.log")
    
    context_list=[]
    for raw_comm in comments:
        user_input=COMMENT_REFINE_TEMPLATE+"\n### Input: \n"+raw_comm
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
    
    res_list=[res.strip() for res in res_list]
    return res_list
    
    
if __name__ =="__main__":
    # convert_anno("test.json")
    None
    