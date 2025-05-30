from gettext import find
import json
import os
import shutil
import sys
import time
import re
import asyncio
from string import Template
from zeno_build.models import lm_config
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
    # print(sys.path)
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


visual_def='''
Below we will provide a segment of human evaluation comments for 'visual quality' of certain AI video, composed of keywords pointing out various issues in the video. 
'visual quality' cares about the video's visual and optical propertities, including 'resolution, overall clarity, local blurriness, smoothness, stability of brightness/contrast, distortion/misalignment, abrupt changes, and any other factors the affect the watching experience'. The keywords written by the annotators are also mostly derived from the above factors.
'''

t2v_def='''
Below we will provide a segment of human evaluation comments for 't2v_alignment' of certain AI video, composed of keywords pointing out various issues in the video. 

The 't2v_alignment' dimension mainly assesses whether the generated video fully and accurately depicts the elements mentioned in the text prompt, such as characters, actions, animals, etc., as well as background, quantity, color, weather, and so on. So the keywords written by annotators sometimes only indicate the elements that are missing from the video.
'''

phy_def='''
Below we will provide a segment of human evaluation comments for 'physical consistency' of certain AI video, composed of keywords pointing out various issues in the video. 

The 'physical consistency' dimension mainly examines whether there are any violations of common sense, physical laws, or any other aspects in the video that appear strange or unnatural. Most of the keywords provided by annotators point out the specific abnormalities or inconsistencies they observed in the video.
'''


refine_template=Template("""
We are collecting and processing human annotations for the quality evaluation of AI-generated videos in text-to-video generation. 
$dim_def

Please expand and polish these keywords into a complete, natural human evaluation comment with appropriate style and length. 

Your response must follow the format below strictly:
{
    "comment": "<extended and refined comment>" (this field is only allowed to be string)
}
DO NOT include any text before or after the dict block

the text prompt used to generate the video: 
$prompt
anno_keywords: 
$comment                
""")


def _video_path(video_dir,video_name):
    return ""

async def _refine_comment_gpt(comments,prompts,template,dim_def):
    
    # model_name="gpt-4o-2024-08-06"
    model_name = "gpt-4o-mini"
    model_config = lm_config.LMConfig(provider="openai_chat", model=model_name)
    
    date_time=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    os.makedirs(f"./logs",exist_ok=True)
    logger=set_logger(f"./logs/import_anno_{date_time}.log")
    
    context_list=[]
    for raw_comment,prompt in zip(comments,prompts):
        context=dict(messages=
            [{
                "content":template.substitute(dim_def=dim_def,prompt=prompt,comment=raw_comment),
                "role":"user"
            }])
        context_list.append(context)
    
    res_list = await generate_from_openai_chat_completion(
        full_contexts = context_list,
        model_config = model_config,
        logger=logger,
    )
    final_list=[]
    for res in res_list:
        try:
            res="{"+res.split("{")[-1].split("}")[0].strip()+"}"
            eval_res = eval(res)
            if isinstance(eval_res, dict) and 'comment' in eval_res:
                final_list.append(eval_res["comment"])
        except Exception as e:
            final_list.append(" ")
    return final_list


def convert_anno(raw_anno_file,save_path):
    with open(raw_anno_file,"r",encoding="utf-8") as f:
        raw_annos=json.load(f)[:3]
        
    data=[]
    prompts=[]
    visual_scores=[]
    visual_comments=[]
    t2v_scores=[]
    t2v_comments=[]
    phy_scores=[]
    phy_comments=[]
    for anno in raw_annos:
        url=anno["info"]["data"][2]["content"]
        video_name=url.split("/")[-1].split(".")[0]
        prompt_en=anno["info"]["data"][1]["content"].split("English Prompt")[1].split("\n")[0].strip(". :\n")
        prompt_cn=anno["info"]["data"][1]["content"].split("翻译为中文的Prompt")[1].split("\n")[0].strip(". :\n")
        prompts.append(prompt_en)
        try:
            visual_score=re.search(r'\d+', str(anno["labels"][0]["data"]["value"])).group()
        except:
            print(f"visual score not found for {video_name}")
            continue
        try:
            t2v_score=re.search(r'\d+', str(anno["labels"][2]["data"]["value"])).group()
        except:
            print(f"t2v score not found for {video_name}")
            continue
        try:
            physical_score=re.search(r'\d+', str(anno["labels"][4]["data"]["value"])).group()
        except:
            print(f"physical score not found for {video_name}")
            continue
        
        visual_comment=anno["labels"][1]["data"]["value"]
        t2v_comment=anno["labels"][3]["data"]["value"]
        phy_comment=anno["labels"][5]["data"]["value"]
        if visual_score==MIN_SCORE or visual_score==MAX_SCORE:
            visual_comment="NA"
        if t2v_score==MAX_SCORE:
            t2v_comment="NA"
        if physical_score==MAX_SCORE:
            phy_comment="NA"     
               
        visual_scores.append(visual_score)
        visual_comments.append(visual_comment)
        t2v_scores.append(t2v_score)
        t2v_comments.append(t2v_comment)
        phy_scores.append(physical_score)
        phy_comments.append(phy_comment)
    
        data_item={
            "video_name":video_name,
            "video_path":_video_path(video_dir,video_name),
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

    print('len of data list: ',len(data))

    visual_comments_refined=asyncio.run(_refine_comment_gpt(visual_comments,prompts,refine_template,dim_def=visual_def))
    t2v_comments_refined=asyncio.run(_refine_comment_gpt(t2v_comments,prompts,refine_template,dim_def=t2v_def))
    physical_comments_refined=asyncio.run(_refine_comment_gpt(phy_comments,prompts,refine_template,dim_def=phy_def))
    
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
        if phy_scores[idx]==MIN_SCORE:
            data[idx]['physical']["comment"]=shared_comments["physical_1"]
        
        
    with open(save_path,"a",encoding="utf-8") as f:
        json.dump(data,f,indent=4,ensure_ascii=False)


    
if __name__ =="__main__":
    data_dir="/data/xuan/videoscore2/data"
    video_dir=""
    all_data_path=f"{data_dir}/all_anno.json"
    shared_comments=json.load(open("./data/const/shared_comments.json","r"))
    MIN_SCORE=1
    MAX_SCORE=5
    os.environ["OPENAI_API_KEY"]=os.environ["DEEPBRICKS_KEY1"]
    os.environ["OPENAI_BASE_URL"]=os.environ["DEEPBRICKS_URL"]
    input_path="data/test.json"
    save_path="data/test_save.json"
    
    convert_anno(input_path,save_path)