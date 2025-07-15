import json
import os
from time import sleep
import asyncio
import ast
from string import Template
from zeno_build.models import lm_config
from utils_async_chat import generate_from_openai_chat_completion,set_logger
from tqdm import tqdm
import argparse


MODIFY_TEMPLATE=Template("""
I'm conducting a multi-dimensional quality assessment of AI-generated videos, focusing on the dimensions of Visual Quality, Text-to-Video Consistency, and Physical Consistency (also referred to as Common-sense Consistency).

In the following I will provide a multi-dimensional analysis of a specific video. However, the scores assigned in the analysis may not be entirely accurate. I will provide the ground-truth scores for each dimension, and your task is to adjust the analysis text accordingly to ensure it aligns with the actual scores. The scale of score is [1, 2, 3, 4, 5].

**Important Notes:**

(1) **Any human comment should NOT be mentioned in the output analysis**. If the input analysis quote or mention human comments, you should pretend not to know them in your output, they are provided solely to inform and enhance your understanding for better evaluation.

(2) **DO NOT** alter the overall structure or core meaning of the analysis. Only revise specific expressions or phrases as needed so that the content reasonably reflects the provided scores. 

(3) **DO NOT** change the length of analysis, your output analysis should be no shorter than the input analysis. If you think the input analysis is not very specific, you can also extend it approximately.

Your response must follow the format below strictly:
{
    'new_thinking': "<modified analysis>" (this field is only allowed to be string),
}
DO NOT include any text before or after the dictionary block.

Here is the input:
multi-dimensional analysis:
$thinking

ground-truth of Dim-1: 'Visual Quality':
$v_score

ground-truth of Dim-2: 'Text-to-Video Consistency':
$t_score

ground-truth of Dim-3: 'Physical Consistency' (also referred to as Common-sense Consistency):
$p_score
                         
""")


async def _bot_modify_thinking(items,model_config,logger):
    context_list=[]
    for x in items:
        v_score=int(x["visual_score"])
        t_score=int(x["t2v_score"])
        p_score=int(x["phy_score"])
        thinking=x["thinking"]
        user_input=MODIFY_TEMPLATE.substitute(thinking=thinking,v_score=v_score,t_score=t_score,p_score=p_score)
        
        context=dict(messages=
                    [{
                        "content":user_input,
                        "role":"user"
                    }]      
                )
        context_list.append(context)
        
    res_list = await generate_from_openai_chat_completion(
        full_contexts = context_list,
        model_config = model_config,
        logger=logger,
        )
    
    for idx,res in enumerate(res_list):
        video_name=items[idx]["video_name"]
        logger.info(f"\n----------------- {video_name} raw output -----------------\n {res}")
        res = "{" + res.split("{")[-1].split("}")[0].strip() + "}"
        try:
            eval_res = ast.literal_eval(str(res))
            items[idx]["thinking"]=eval_res["new_thinking"]
        except Exception as e:
            print(e)
            items[idx]["thinking"]=None
            continue
        
    return items


def _difference_is_2(s1,s2,dim_name):
    # s1: human_score
    # s2: model_score
    return int((s1+s2)/2)
    
    # if [s1,s2]==[5,3]:
    #     return 4
    # if [s1,s2]==[3,5]:
    #     return 4
    # if [s1,s2]==[4,2]:
    #     return 3
    # if [s1,s2]==[2,4]:
    #     return 2
    # if [s1,s2]==[3,1]:
    #     return 2
    # if [s1,s2]==[1,3]:
    #     return 1


def _difference_is_1(s1,s2,dim_name):
    # s1: human_score
    # s2: model_score
    if dim_name in ["visual","t2v"]:
        return s1
    
    elif dim_name == "phy":
        if [s1,s2]==[5,4] or [s1,s2]==[4,5]:
            return 4
        else:
            return s1
    
    # if [s1,s2]==[4,3]:
    #     return 4
    # if [s1,s2]==[3,4]:
    #     return 3
    # if [s1,s2]==[3,2]:
    #     return 3
    # if [s1,s2]==[2,3]:
    #     return 2
    # if [s1,s2]==[2,1]:
    #     return 2
    # if [s1,s2]==[1,2]:
    #     return 1



async def modify_score_thinking(src_paths,save_path,rej_path):
    data=[]
    for src_path in src_paths:
        with open(src_path,"r",encoding='utf-8') as f:
            data.extend(json.load(f))
    
    new_data=[]
    skip_num=0
    diff2_num=0
    diff1_num=0
    diff0_num=0
    model_high_num=0
    model_low_num=0
    
    modify_needed_items=[]
    skipped_items=[]
    for idx,item in tqdm(enumerate(data)):
        video_name=item["video_name"]
        v_score=int(item["visual_score"])
        v_score_model=int(item["visual_score_model"])
        t_score=int(item["t2v_score"])
        t_score_model=int(item["t2v_score_model"])
        p_score=int(item["phy_score"])
        p_score_model=int(item["phy_score_model"])
        score_modified=False
        item_skipped=False
        
        new_item=item
        for dim_name, human_score, model_score in zip(["visual","t2v","phy"],
                                                      [v_score,t_score,p_score],
                                                      [v_score_model,t_score_model,p_score_model]):
                
            if abs(human_score-model_score)>=3:
                skip_num+=1
                item_skipped=True
            
            if abs(human_score-model_score)==2:
                new_score=_difference_is_2(human_score,model_score,dim_name)  
                # new_score=human_score    
                # new_score=model_score   
                new_item[f"{dim_name}_score"]=new_score
                diff2_num+=1
                
            if abs(human_score-model_score)==1:
                new_score=_difference_is_1(human_score,model_score,dim_name)
                # new_score=human_score
                # new_score=model_score
                new_item[f"{dim_name}_score"]=new_score
                diff1_num+=1
                
                if human_score < model_score:
                    model_high_num+=1
                if human_score > model_score:
                    model_low_num+=1
            
            if abs(human_score-model_score)==0:
                new_item[f"{dim_name}_score"]=model_score 
                diff0_num+=1
            
            if new_item[f"{dim_name}_score"]!=model_score:
                score_modified=True
        
        if item_skipped==True:
            skipped_items.append(item)
            continue
        
        new_item.pop("visual_score_model",None)
        new_item.pop("t2v_score_model",None)
        new_item.pop("phy_score_model",None)
        new_item.pop("visual_cmt_raw",None)
        new_item.pop("t2v_cmt_raw",None)
        new_item.pop("phy_cmt_raw",None)
        if score_modified==True:
            modify_needed_items.append(new_item)
            
        new_data.append(new_item)
    
    
    logger=set_logger(logger_file=log_path)
    print("1st round of modify: ", len(modify_needed_items))
    modified_items=await _bot_modify_thinking(modify_needed_items,MODEL_CONFIG,logger)
    for m_x in modified_items:
        for idx,_ in enumerate(new_data):
            if new_data[idx]["video_name"]==m_x["video_name"] and m_x['thinking'] is not None:
                new_data[idx]=m_x
    
    modify_needed_items=[x for x in modified_items if x['thinking'] is None]
    print("2nd round of modify: ", len(modify_needed_items))
    modified_items=await _bot_modify_thinking(modify_needed_items,MODEL_CONFIG,logger)
    for m_x in modified_items:
        for idx,_ in enumerate(new_data):
            if new_data[idx]["video_name"]==m_x["video_name"] and m_x['thinking'] is not None:
                new_data[idx]=m_x
    
    modify_needed_items=[x for x in modified_items if x['thinking'] is None]
    print("Remained error items: ", len(modify_needed_items))
    
    with open(save_path,"w") as f:
        json.dump(new_data,f,indent=4,ensure_ascii=False)
        
    with open(rej_path,"w") as f:
        json.dump(skipped_items,f,indent=4,ensure_ascii=False)
    
    
    # from _analyze import plot
    # v_scores=[x['visual_score'] for x in new_data]
    # t_scores=[x['t2v_score'] for x in new_data]
    # p_scores=[x['phy_score'] for x in new_data]
    # plot(v_scores,batch_name,1)
    # plot(t_scores,batch_name,2)
    # plot(p_scores,batch_name,3)    
    
    # print(">=3",skip_num)
    # print("2",diff2_num)
    # print("1",diff1_num)
    # print("0",diff0_num)
    # print("\n")
    # print("model>human",model_high_num)
    # print("model<low",model_low_num)
    
    
    
if __name__ == "__main__":
    # cd VideoScore2/data
    # tmux new-session -s run0
    # conda activate base
    # python modify_score_thinking.py --run_idx=0
    
    
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_idx", required=True)
    args = ap.parse_args()
    run_idx=args.run_idx
    
    src_paths=[
        f"thinking_split/thinking_17k_{run_idx}.json",
        # "thinking_cmt/sft_17k_modified.json",
        # "thinking_cmt/thinking_com_5k_original.json"
    ]
    batch_name=f"sft_17k_{run_idx}"
    
    save_dir="thinking_final"
    os.makedirs(save_dir,exist_ok=True)
    save_path=os.path.join(save_dir,f"final_{batch_name}.json")
    rej_dir="thinking_rejected"
    os.makedirs(rej_dir,exist_ok=True)
    rej_path=os.path.join(rej_dir,f"rej_{batch_name}.json")
    
    log_path="modify_logs/test.log"
    if int(run_idx)>=9:
        os.environ["OPENAI_API_KEY"]=os.environ[f"DEEPBRICKS_KEY1"]
    else:
        os.environ["OPENAI_API_KEY"]=os.environ[f"DEEPBRICKS_KEY{run_idx}"]
    os.environ["OPENAI_BASE_URL"]=os.environ["DEEPBRICKS_URL"]

    model_name='gpt-4o-mini'
    MODEL_CONFIG= lm_config.LMConfig(provider="openai_chat", model=model_name)
    MAX_SCORE=5
    asyncio.run(modify_score_thinking(src_paths,save_path,rej_path))