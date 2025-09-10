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
I'm conducting a multi-dimensional quality assessment of AI-generated videos, focusing on the dimensions of (1) Visual Quality, (2) Text-to-Video Consistency, and (3) Physical/Common-sense Consistency.

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


async def _bot_edit_thinking(items,model_config,logger):
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
    if dim_name in ["visual","phy"]:
        return s1
    
    elif dim_name == "t2v":
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



async def align_thinking_and_score(src_path,save_path,rej_path,log_path):
    data=[]
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
        # new_item.pop("visual_cmt_raw",None)
        # new_item.pop("t2v_cmt_raw",None)
        # new_item.pop("phy_cmt_raw",None)
        if score_modified==True:
            modify_needed_items.append(new_item)
            
        new_data.append(new_item)
    
    
    logger=set_logger(logger_file=log_path)
    
    for round_idx in range(MODIFY_ROUND_NUM):
        if len(modify_needed_items)==0:
            break
        logger.info(f"modify round {round_idx}, items: ", len(modify_needed_items))
        modified_items=await _bot_edit_thinking(modify_needed_items,MODEL_CONFIG,logger)
        for m_x in modified_items:
            for idx,_ in enumerate(new_data):
                if new_data[idx]["video_name"]==m_x["video_name"] \
                    and m_x['thinking'] is not None:
                    new_data[idx]=m_x
        modify_needed_items=[x for x in modified_items if (x['thinking'] is None)]
    
    logger.info("Remained error items: ", len(modify_needed_items))
    
    new_data=[x for x in new_data if x['thinking'] is not None]
    logger.info("Saved items: ",len(new_data))
    
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
    
    # ap = argparse.ArgumentParser()
    # ap.add_argument("--run_idx", required=True)
    # args = ap.parse_args()
    # run_idx=args.run_idx
    
    api_key_idx=3
    batch_names=[
        # 1,2,3,4,5,  #0
        # 13,14,15,17,18,   #0
        # 19,20,21,22,23,   #1
        # 24,29,30,31,32,   #1
        # 53,54,55,61,69,70,   #2
        "com_5k_0",   #3
        "com_5k_1",   #3
        
        # 9,16,33,34,   #0
        # 38,45,46,47,48,    #4
        # 56,62,71,74,    #5
        # 75,78,79,81,    
        # 82,83,85,86, 
    ]
    src_dir="thinking_new_score"
    save_dir="thinking_final"
    rej_dir="thinking_rej"
    os.makedirs(save_dir,exist_ok=True)
    os.makedirs(rej_dir,exist_ok=True)
    
    model_name='gpt-4.1-mini'
    if int(api_key_idx)>=9:
        os.environ["OPENAI_API_KEY"]=os.environ[f"DEEPBRICKS_KEY1"]
    else:
        os.environ["OPENAI_API_KEY"]=os.environ[f"DEEPBRICKS_KEY{api_key_idx}"]
    os.environ["OPENAI_BASE_URL"]=os.environ["DEEPBRICKS_URL"]
    MODEL_CONFIG= lm_config.LMConfig(provider="openai_chat", model=model_name)
    MAX_SCORE=5
    MODIFY_ROUND_NUM=5
    
    for batch_name in batch_names:
        src_path=os.path.join(src_dir,f"tk_new_score_{batch_name}.json")
        save_path=os.path.join(save_dir,f"final_{batch_name}.json")
        rej_path=os.path.join(rej_dir,f"rej_{batch_name}.json")
        log_path=f"modify_logs/align_thinking_{model_name}_{batch_name}.log"
        
        asyncio.run(align_thinking_and_score(src_path,save_path,rej_path,log_path))
        
        
        
        # 9,16,33,34,
        # 38,45,46,47,48,
        # 56,62,71,74,
        # 75,78,79,81,
        # 82,83,85,86,
        
        # 1,2,3,4,5,  #0
        # 13,14,15,17,18,   #0
        # 19,20,21,22,23,   #1
        # 24,29,30,31,32,   #1
        # 53,54,55,61,69,70,   #2
        # "com_5k_0",   #3
        # "com_5k_1",   #3
        # "com_5k_2",   #4
        # "com_5k_3",   #4
        # "com_5k_4",   #5