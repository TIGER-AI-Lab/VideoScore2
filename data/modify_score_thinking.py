import json
import os
from time import sleep
import asyncio
import ast
from string import Template
from zeno_build.models import lm_config
from utils_async_chat import generate_from_openai_chat_completion,set_logger


MODIFY_TEMPLATE=Template("""
I'm conducting a multi-dimensional quality assessment of AI-generated videos, focusing on the dimensions of Visual Quality, Text-to-Video Consistency, and Physical Consistency (also referred to as Common-sense Consistency).

In the following I will provide a multi-dimensional analysis of a specific video. However, the scores assigned in the analysis may not be entirely accurate. I will provide the ground-truth scores for each dimension, and your task is to adjust the analysis text accordingly to ensure it aligns with the actual scores.

**Important Notes**:
Do not alter the overall structure or core meaning of the analysis. Do not change the approximate length of analysis. Only revise specific expressions or phrases as needed so that the content reasonably reflects the provided scores.

Your response must follow the format below strictly:
{
    'new_thinking': "<modified thinking process>" (this field is only allowed to be string),
}
DO NOT include any text before or after the dict block

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


    
def modify_score_thinking(src_path,save_path):
    with open(src_path,"r",encoding='utf-8') as f:
        data=json.load(f)
        
    data=data[1000:2000]
    
    new_data=[]
    
    skip_num=0
    avg_num=0
    cover_num=0
    keep_num=0
    
    model_high_num=0
    model_equal_num=0
    model_low_num=0
    
    modify_needed_items=[]
    skipped_items=[]
    for idx,item in enumerate(data):
        video_name=item["video_name"]
        v_score=int(item["visual_score"])
        v_score_model=int(item["visual_score_model"])
        t_score=int(item["t2v_score"])
        t_score_model=int(item["t2v_score_model"])
        p_score=int(item["phy_score"])
        p_score_model=int(item["phy_score_model"])
        score_modified=False
        
        new_item=item
        new_item.pop("visual_score_model",None)
        new_item.pop("t2v_score_model",None)
        new_item.pop("phy_score_model",None)
        new_item.pop("visual_cmt_raw",None)
        new_item.pop("t2v_cmt_raw",None)
        new_item.pop("phy_cmt_raw",None)
        
        for dim_name, human_score, model_score in zip(["visual","t2v","phy"],
                                                      [v_score,t_score,p_score],
                                                      [v_score_model,t_score_model,p_score_model]):
                
            if abs(human_score-model_score)>=3:
                skip_num+=3
                skipped_items.append(item)
            
            if abs(human_score-model_score)==2:
                new_score=int((human_score+model_score)/2)              
                new_item[f"{dim_name}_score"]=new_score
                score_modified=True
                avg_num+=1
                
            if abs(human_score-model_score)==1 and MAX_SCORE in [human_score,model_score]:
                new_score=min(human_score,model_score) 
                new_item[f"{dim_name}_score"]=new_score
                if new_score!=model_score:
                    score_modified=True
                cover_num+=1
                
                if human_score < model_score:
                    model_high_num+=1
                if human_score > model_score:
                    model_low_num+=1
            
            if abs(human_score-model_score)==1 and MAX_SCORE not in [human_score,model_score]:
                new_item[f"{dim_name}_score"]=model_score 
                cover_num+=1
                if human_score < model_score:
                    model_high_num+=1
                if human_score > model_score:
                    model_low_num+=1
                
            if abs(human_score-model_score)==0:
                new_item[f"{dim_name}_score"]=model_score 
                keep_num+=1
            
        new_data.append(new_item)
        if score_modified==True:
            modify_needed_items.append(new_item)
    
    print(len(modify_needed_items))
    
    # modified_items=asyncio.run(_bot_modify_thinking(modify_needed_items,MODEL_CONFIG,logger))
    
    # for idx,_ in enumerate(new_data):
    #     for m_x in modified_items:
    #         if new_data[idx]["video_name"]==m_x["video_name"]:
    #             new_data[idx]=m_x
            
    # with open(save_path,"w") as f:
    #     json.dump(new_data,f,indent=4,ensure_ascii=False)
        
    from _analyze import plot
    batch_name="modified_com_5k"
    v_scores=[x['visual_score'] for x in new_data]
    t_scores=[x['t2v_score'] for x in new_data]
    p_scores=[x['phy_score'] for x in new_data]
    plot(v_scores,batch_name,1)
    plot(t_scores,batch_name,2)
    plot(p_scores,batch_name,3)    
    
    # print(">=3",skip_num)
    # print("==2",avg_num)
    # print("==1",cover_num)
    # print("==0",keep_num)
    # print("\n")
    # print("model>human",model_high_num)
    # print("model=human",model_equal_num)
    # print("model<low",model_low_num)
    
if __name__ == "__main__":
    # src_path="_prev/thinking/allin1_2shot/res_claude-sonnet-4-20250514.json"
    src_path="thinking_cmt/thinking_com_5k.json"
    
    save_dir="thinking_final"
    os.makedirs(save_dir,exist_ok=True)
    save_path=os.path.join(save_dir,f"final_{src_path.split('/')[1].split('thinking_')[1]}")
    
    rej_dir="thinking_rejected"
    os.makedirs(rej_dir,exist_ok=True)
    rej_path=os.path.join(rej_dir,f"rej_{src_path.split('/')[1].split('thinking_')[1]}")
    
    log_path="modify_logs/test.log"
    logger=set_logger(logger_file=log_path)
    
    os.environ["OPENAI_API_KEY"]=os.environ["DEEPBRICKS_KEY1"]
    os.environ["OPENAI_BASE_URL"]=os.environ["DEEPBRICKS_URL"]

    model_name='gpt-4o-mini'
    MODEL_CONFIG= lm_config.LMConfig(provider="openai_chat", model=model_name)
    MAX_SCORE=5
    modify_score_thinking(src_path,save_path)