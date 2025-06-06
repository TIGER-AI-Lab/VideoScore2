import json
import os
import re
import asyncio
import argparse
from string import Template
from refine_cmt_async_gpt import _refine_cmt_async_gpt
from refine_cmt_claude import _refine_cmt_claude
from refine_cmt_gemini import _refine_cmt_gemini
from refine_cmt_gpt import _refine_cmt_gpt


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

def _hf_folder_size(repo_id,folder):
    from huggingface_hub import list_repo_files
    all_files = list_repo_files(repo_id=repo_id,repo_type="dataset")
    files = set()

    for f in all_files:
        if f.startswith(f"{folder}/"):
            parts = f[len(folder) + 1:].split("/")
            if parts:
                files.add(parts[0])
    return len(files)

def _hf_file_exist(repo_id,target_file):
    from huggingface_hub import list_repo_files
    all_files = list_repo_files(repo_id=repo_id,repo_type="dataset")
    return target_file in all_files
        

def convert_anno(anno_path,save_path,num,model_name,model_access,append_img):
    with open(anno_path,"r",encoding="utf-8") as f:
        if type(num) is int:
            raw_annos=json.load(f)[:num]
        else:
            raw_annos=json.load(f)
            
    data=[]
    prompts=[]
    visual_scores=[]
    visual_cmts=[]
    t2v_scores=[]
    t2v_cmts=[]
    phy_scores=[]
    phy_cmts=[]
    frames_2d_list=[]
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
            phy_score=re.search(r'\d+', str(anno["labels"][4]["data"]["value"])).group()
        except:
            print(f"physical score not found for {video_name}")
            continue
        
        visual_cmt=anno["labels"][1]["data"]["value"]
        t2v_cmt=anno["labels"][3]["data"]["value"]
        phy_cmt=anno["labels"][5]["data"]["value"]
        if visual_score==MIN_SCORE or visual_score==MAX_SCORE:
            visual_cmt="NA"
        if t2v_score==MAX_SCORE:
            t2v_cmt="NA"
        if phy_score==MAX_SCORE:
            phy_cmt="NA"     
               
        visual_scores.append(visual_score)
        visual_cmts.append(visual_cmt)
        t2v_scores.append(t2v_score)
        t2v_cmts.append(t2v_cmt)
        phy_scores.append(phy_score)
        phy_cmts.append(phy_cmt)
        if append_img:
            # format of frames path: <video_frames_dir>/frames/<video_name>_<frame_idx>.jpg
            try:
                n_frames=_hf_folder_size(REPO_ID,video_name)
                if not all(_hf_file_exist(REPO_ID,f"{video_name}/{video_name}_{i}.jpg") for i in range(n_frames)):
                    print("not all frames exists, skipped")
                    continue
            except Exception as e:
                print(e, "\nerror in fetch video frames.")
                continue
            
            frames_2d_list.append([f"{IMG_HF_PREFIX}/{video_name}/{video_name}_{i}.jpg" for i in range(n_frames)])

            
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
                "score":phy_score,
                "comment":None,
            }
        }
        data.append(data_item)

    print('len of data list: ',len(data))
        
    # visual_cmts_refined=asyncio.run(_refine_cmt_async_gpt(
    #     visual_cmts,prompts,frames_2d_list,refine_template,visual_def))
    # t2v_cmts_refined=asyncio.run(_refine_cmt_async_gpt(
    #     t2v_cmts,prompts,frames_2d_list,refine_template,t2v_def))
    # phy_cmts_refined=asyncio.run(_refine_cmt_async_gpt(
    #     phy_cmts,prompts,frames_2d_list,refine_template,phy_def))
    
    if "gpt" in model_name:
        visual_cmts_refined=_refine_cmt_gpt(
            model_name,model_access,visual_cmts,prompts,frames_2d_list,refine_template,visual_def)
        t2v_cmts_refined=_refine_cmt_gpt(
            model_name,model_access,t2v_cmts,prompts,frames_2d_list,refine_template,t2v_def)
        phy_cmts_refined=_refine_cmt_gpt(
            model_name,model_access,phy_cmts,prompts,frames_2d_list,refine_template,phy_def)
        
    elif "gemini" in model_name:
        visual_cmts_refined=_refine_cmt_gemini(
            model_name,model_access,visual_cmts,prompts,frames_2d_list,refine_template,visual_def)
        t2v_cmts_refined=_refine_cmt_gemini(
            model_name,model_access,t2v_cmts,prompts,frames_2d_list,refine_template,t2v_def)
        phy_cmts_refined=_refine_cmt_gemini(
            model_name,model_access,phy_cmts,prompts,frames_2d_list,refine_template,phy_def)
    
    elif "claude" in model_name:
        visual_cmts_refined=_refine_cmt_claude(
            model_name,model_access,visual_cmts,prompts,frames_2d_list,refine_template,visual_def)
        t2v_cmts_refined=_refine_cmt_claude(
            model_name,model_access,t2v_cmts,prompts,frames_2d_list,refine_template,t2v_def)
        phy_cmts_refined=_refine_cmt_claude(
            model_name,model_access,phy_cmts,prompts,frames_2d_list,refine_template,phy_def)
    else:
        print("model not supported, exited")
        exit()
    
    for idx in range(len(visual_scores)):     
        data[idx]['visual']["comment"]=visual_cmts_refined[idx]
        data[idx]['t2v_align']["comment"]=t2v_cmts_refined[idx]
        data[idx]['physical']["comment"]=phy_cmts_refined[idx]
        if visual_scores[idx]==MIN_SCORE:
            data[idx]['visual']["comment"]=shared_cmts["visual_1"]
        if visual_scores[idx]==MAX_SCORE:
            data[idx]['visual']["comment"]=shared_cmts["visual_5"]
        if t2v_scores[idx]==MAX_SCORE:
            data[idx]['t2v_align']["comment"]=shared_cmts["t2v_5"]
        if phy_scores[idx]==MAX_SCORE:
            data[idx]['physical']["comment"]=shared_cmts["phy_5"]
        
        
    with open(save_path,"a",encoding="utf-8") as f:
        json.dump(data,f,indent=4,ensure_ascii=False)



if __name__ =="__main__":
    shared_cmts={
        "visual_5": "",
        "t2v_5": "",
        "phy_5": "",
        "visual_1": ""
    }
    MIN_SCORE=1
    MAX_SCORE=5
    REPO_ID="hexuan21/VS2_frame_part_cache"
    IMG_HF_PREFIX=f"https://huggingface.co/datasets/hexuan21/VS2_frame_part_cache/resolve/main"
    
    current_dir=os.path.dirname(os.path.abspath(__file__))
    root_frames_dir=os.path.join(current_dir,"video_frames")
    os.makedirs(root_frames_dir,exist_ok=True)
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--anno_path', type=str, required=True, default="test.json")
    parser.add_argument('--model_name', type=str, required=True, default='gpt-4o-mini')
    parser.add_argument('--append_img', type=bool, required=True, default=True)
    parser.add_argument('--api_key', type=str, required=True,)
    parser.add_argument('--basr_url', type=str, required=False,)

    args = parser.parse_args()
          
    model_name=args.model_name
    model_access={
        "api_key":args.api_key,
        "base_url":args.basr_url,      # only gpt series need this field
    } 
    
    save_path=os.path.join("converted_anno",f"res_{model_name}.json")
    os.makedirs(os.path.dirname(save_path),exist_ok=True)
    convert_anno(anno_path=args.anno_path,save_path=save_path,num="all",
                 model_name=args.model_name,model_access=model_access,
                 append_img=args.append_img)