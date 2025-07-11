import json
import os
from time import sleep


def _bot_modify_thinking(thinking,ref_score):
    while True:
        if num_try >= 3:
            print(f"modify thinking failed")
            return None
        try:
            completion = ()
            
            break
        except Exception as e:
            print(e)
            print(f"modify thinking seems to be wrong, sleep for some time")
            num_try += 1
            sleep(60)


def modify_score_thinking(src_path,save_path):
    with open(src_path,"r",encoding='utf-8') as f:
        data=json.load(f)
        
    data=data[:1000]
    
    new_data=[]
    skip_num=0
    avg_num=0
    cover_num=0
    keep_num=0
    
    model_high_num=0
    model_equal_num=0
    model_low_num=0
    for idx,item in enumerate(data):
        v_score=int(item["visual_score"])
        v_score_model=int(item["visual_score_model"])
        t_score=int(item["t2v_score"])
        t_score_model=int(item["t2v_score_model"])
        p_score=int(item["phy_score"])
        p_score_model=int(item["phy_score_model"])
        thinking=item["thinking"]
        
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
            # if human_score>model_score:
            #     model_low_num+=1
            # if human_score==model_score:
            #     model_equal_num+=1
            # if human_score<model_score:
            #     model_high_num+=1
                
            if abs(human_score-model_score)>=3:
                skip_num+=3
                continue
            
            if abs(human_score-model_score)==2:
                new_score=int((human_score+model_score)/2)
                # try:
                #     new_thinking=_bot_modify_thinking(thinking,new_score)
                # except Exception as e:
                #     print(e)
                #     continue
                # new_item['thinking']=new_thinking
                
                new_item[f"{dim_name}_score"]=new_score
                avg_num+=1
                
            if abs(human_score-model_score)==1 and MAX_SCORE in [human_score,model_score]:
                new_score=min(human_score,model_score) 
                # try:
                #     new_thinking=_bot_modify_thinking(thinking,new_score)
                # except Exception as e:
                #     print(e)
                #     continue
                # new_item['thinking']=new_thinking
                new_item[f"{dim_name}_score"]=new_score
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
                new_item[f"{dim_name}_score"]=human_score 
                keep_num+=1
            
        new_data.append(new_item)
        
        # if abs(v_score-v_score_model)>=3 or abs(t_score-t_score_model)>=3 or abs(p_score-p_score_model)>=3:
        #     skip_num+=1
        #     continue
        # if abs(v_score-v_score_model)==2:
        #     new_item["visual_score"]=int((v_score+v_score_model)/2)    
        #     avg_num+=1
        # if abs(v_score-v_score_model)==1 and MAX_SCORE not in [v_score,v_score_model]:
        #     new_item["visual_score"]=v_score_model 
        #     cover_num+=1
        # if abs(v_score-v_score_model)==1 and MAX_SCORE in [v_score,v_score_model]:
        #     new_item["visual_score"]=min(v_score,v_score_model) 
        #     cover_num+=1

    # with open(save_path,"w") as f:
    #     json.dump(new_data,f,indent=4,ensure_ascii=False)
        
    print(">=3",skip_num)
    print("==2",avg_num)
    print("==1",cover_num)
    print("==0",keep_num)
    print("\n")
    print("model>human",model_high_num)
    print("model=human",model_equal_num)
    print("model<low",model_low_num)
    
if __name__ == "__main__":
    # src_path="_prev/thinking/allin1_2shot/res_claude-sonnet-4-20250514.json"
    src_path="thinking_cmt/thinking_batch_91_100_com.json"
    save_path=""
    MAX_SCORE=5
    modify_score_thinking(src_path,save_path)