import json
import os



def modify_thinking(src_path,save_path):
    with open(src_path,"r",encoding='utf-8') as f:
        data=json.load(f)

    for x in data:
        video_name=x['video_name']
        prompt=x['prompt']
        v_score=x["visual_score"]
        v_score_model=x["visual_score_model"]
        t_score=x["t2v_score"]
        t_score_model=x["t2v_score_model"]
        p_score=x["phy_score"]
        p_score_model=x["phy_score_model"]
        thinking=x["thinking"]
        
        
    
    
    
if __name__ == "__main__":
    src_path=""
    save_path=""
    modify_thinking(src_path,save_path)