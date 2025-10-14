import json
import os
import argparse
import time
import re
from tqdm import tqdm
from benchmark import VS2_QUERY_TEMPLATE,VS1_REG_QUERY_TEMPLATE,AIGVE_MACS_QUERY_TEMPLATE,DIM_NAMES,load_benchmark
from string import Template

def main(args):
    method=args.method
    bench=args.bench
    bench_data_num=args.bench_data_num
    model_name_or_path=args.model_name_or_path
    kwargs = json.loads(args.kwargs)
        
    if isinstance(eval(bench_data_num),int):
        bench_data_num=eval(bench_data_num)

    bench_data=load_benchmark(bench_data_dir,bench,bench_data_num)
    
    print("benchmark data loaded.")
    
    if method.lower() == "vs2":
        ## conda activate vs2_eval
        from eval_methods.vs2 import eval_VideoScore2
        model=eval_VideoScore2(model_name_or_path)    
        q_template=VS2_QUERY_TEMPLATE
    
    elif method.lower() == "vs2_float":
        ## conda activate vs2_eval
        from eval_methods.vs2_float import eval_VideoScore2_float
        model=eval_VideoScore2_float(model_name_or_path)    
        q_template=VS2_QUERY_TEMPLATE
    
    elif method.lower() == "vs2_float_weighted":
        ## conda activate vs2_eval
        from eval_methods.vs2_float_weighted import eval_VideoScore2_float
        model=eval_VideoScore2_float(model_name_or_path)    
        q_template=VS2_QUERY_TEMPLATE
    
    elif method.lower() == "vs1":
        ## conda activate vs1_eval
        from eval_methods.vs1 import eval_VideoScore1
        model_name_or_path="TIGER-Lab/VideoScore"
        model=eval_VideoScore1(model_name_or_path) 
        q_template=VS1_REG_QUERY_TEMPLATE
    
    elif method.lower() == "unified_reward":
        ## conda activate unifiedreward
        from eval_methods.unified_reward import eval_UnifiedReward
        model_name_or_path="CodeGoat24/UnifiedReward-7b"
        model=eval_UnifiedReward(model_name_or_path) 
        q_template=VS2_QUERY_TEMPLATE
    
    elif method.lower() == "video_reward":
        ## conda activate video_reward
        from eval_methods.video_reward import eval_VideoReward
        model_name_or_path="./eval_methods/utils_video_reward/checkpoints/VideoReward"
        model=eval_VideoReward(model_name_or_path) 
        q_template=Template("""$t2v_prompt""")
    
    elif method.lower() == "vision_reward":
        ## conda activate vision_reward
        from eval_methods.vision_reward import eval_VisionReward
        model_name_or_path="THUDM/VisionReward-Video"
        model=eval_VisionReward(model_name_or_path) 
        q_template=Template("""$t2v_prompt""")
    
    elif method.lower() == "image_reward":
        ## conda activate image_reward
        from eval_methods.image_reward import eval_ImageReward
        model_name_or_path="ImageReward-v1.0"
        model=eval_ImageReward(model_name_or_path) 
        q_template=Template("""$t2v_prompt""")
    
    elif method.lower() == "aigve_macs":
        ## conda activate vs2_eval
        from eval_methods.aigve_macs import eval_AIGVE_MACS
        model_name_or_path="xiaoliux/AIGVE-MACS"
        model=eval_AIGVE_MACS(model_name_or_path) 
        q_template=AIGVE_MACS_QUERY_TEMPLATE
    
    elif method.lower() == "video_phy2_auto_eval":
        ## conda activate videophy
        from eval_methods.video_phy2 import eval_VideoPhy2
        model_name_or_path="./eval_methods/utils_video_phy2/checkpoints/videophy_2_auto" 
        model=eval_VideoPhy2(model_name_or_path) 
        q_template=Template("""$t2v_prompt""")
    
    elif method.lower() == "lift":
        ## conda activate lift
        from eval_methods.lift import eval_LiFT
        model_name_or_path="Fudan-FUXI/LiFT-Critic-13b-lora-v1.5"
        # model_name_or_path="Fudan-FUXI/LiFT-Critic-40b-lora-v1.5"
        model=eval_LiFT(model_name_or_path) 
        q_template=Template("""$t2v_prompt""")
    
    elif method.lower() == "dover":
        ## conda activate dover
        from eval_methods.dover import eval_DOVER
        model_name_or_path="dover"
        model=eval_DOVER() 
        q_template=Template("""$t2v_prompt""")
    
    elif method.lower() == "q_insight":
        ## conda activate q_insight
        from eval_methods.q_insight import eval_Q_Insight
        model_name_or_path="ByteDance/Q-Insight"
        model=eval_Q_Insight(model_name_or_path) 
        q_template=Template("""$t2v_prompt""")
    
    elif method.lower() == "q_align":
        ## conda activate q_align
        from eval_methods.q_align import eval_Q_Align
        model_name_or_path="Q-Align"
        model=eval_Q_Align() 
        q_template=Template("""$t2v_prompt""")
    
    elif method.lower() == "deqa":
        ## conda activate deqa
        from eval_methods.deqa import eval_DeQA
        model_name_or_path="zhiyuanyou/DeQA-Score-Mix3"
        model=eval_DeQA(model_name_or_path) 
        q_template=Template("""$t2v_prompt""")
    
    else:
        print("model not supported")
        exit()
        
    if '/' in model_name_or_path:
        model_name_or_path=model_name_or_path.split('/')[-1]
    eval_res_path=f"res_data/res_{bench}/{model_name_or_path}.json"
    if method in ["vs2","vs2_float","vs2_float_weighted"]:
        infer_fps=kwargs.get("infer_fps",2.0)
        if isinstance(infer_fps,str):
            if infer_fps != "raw" :
                raise Exception("[error] Arg 'infer fps' has invalid type!")                
        elif isinstance(infer_fps,int) or isinstance(infer_fps,float):
            infer_fps=int(infer_fps)
        else:
            raise Exception("[error] Arg 'infer fps' has invalid type!") 
        eval_res_path=f"res_data/res_{bench}/{model_name_or_path}_infer_{infer_fps}fps.json"
    
    if method == "vs2_float":
        temperature=kwargs.get("temperature",0.7)
        eval_res_path=eval_res_path.replace(".json",f"_flt_normed_tempe={temperature}.json")
    
    if method == "vs2_float_weighted":
        temperature=kwargs.get("temperature",0.7)
        eval_res_path=eval_res_path.replace(".json",f"_flt_weighted_tempe={temperature}.json")
    
    print("Evaluation result will be saved to ", eval_res_path)
    
    res_data=[]
    if not os.path.exists(eval_res_path):
        with open(eval_res_path,"w",encoding='utf-8') as f:
            json.dump(res_data,f,indent=4,ensure_ascii=False)
    else:
        with open(eval_res_path,"r") as f:
            res_data=json.load(f)
        dedup_video_names=set()
        dedup_res_data=[]
        for item in res_data:
            if item['video_name'] not in dedup_video_names:
                dedup_res_data.append(item)
                dedup_video_names.add(item['video_name'])
        res_data=dedup_res_data
        with open(eval_res_path,"w",encoding='utf-8') as f:
            json.dump(res_data,f,indent=4,ensure_ascii=False)
        
        print(f"Loaded existing {len(res_data)} res items for bench:{bench}, method:{method}")
        print("Remaining items to eval: ", len(bench_data)-len(res_data))
        bench_data=[item for item in bench_data if item['video_name'] not in set([x['video_name'] for x in res_data])]
        
        
    metrics_report_path=f"metrics_report/met_{bench}/{model_name_or_path}.json"
    os.makedirs(os.path.dirname(eval_res_path),exist_ok=True)
    os.makedirs(os.path.dirname(metrics_report_path),exist_ok=True)
        
        
    for item in tqdm(bench_data):
        video_name=item['video_name']
        prompt=item['prompt']
        path_or_url=os.path.abspath(f"{bench_data_dir}/{bench}/videos/{video_name}.mp4")
        res_item=item
        try:
            user_prompt=q_template.substitute(t2v_prompt=prompt)
            s_t=time.time()
            v_out, t_out, p_out, raw_output = model.evaluate_video(user_prompt, path_or_url, kwargs)
            print("time cost: ",time.time()-s_t)
            print("out:", v_out, t_out, p_out)
        
        except Exception as e:
            print(f"{e}\nerror in evaluation, skipped {video_name}")
            continue
           
        if "vs2" in bench \
            or bench in ["aigve_bench","aigve-bench",
                         "video_phy","video_phy_test_public",
                         "video_phy2","video_phy2_test",
                         "mj_video_bench","mj_bench_video","mj-video-bench","mj-bench-video",
                         "t2vqa_db",
                         "tvge"]:  
                
            if "vs2" in bench:    
                v_gt=item['visual_score']
                t_gt=item['t2v_score']
                p_gt=item['phy_score']
            
            elif bench in ["aigve_bench","aigve-bench"]:
                v_gt=int(round((item['technical_quality']+item['element_quality']+item['action_quality'])/3))
                t_gt=int(round((item['element_presence']+item['action_presence'])/2))
                p_gt=item['physics']
            
            elif bench in ["video_phy","video_phy_test_public",
                           "video_phy2","video_phy2_test",]:
                v_gt=None
                t_gt=item['semantic']
                p_gt=item['physical']
            
            elif bench in ["mj_video_bench","mj_bench_video","mj-video-bench","mj-bench-video"]:
                v_gt=item['fineness']
                t_gt=item['alignment']
                p_gt=item['consistency']
                
            elif bench in ["tvge"]:
                v_gt=item['video_quality_score']
                t_gt=item['text_alignment_score']
                p_gt=None
            
            elif bench in ["t2vqa_db"]:
                v_gt=t_gt=p_gt=item['quality_score']
            
            print(f"gt: {v_gt} {t_gt} {p_gt}")
            res_item.update({
                "v_score_gt":v_gt, "t_score_gt":t_gt, "p_score_gt":p_gt,
                "v_score_model":v_out, "t_score_model":t_out, "p_score_model":p_out,
                "output":raw_output
            })
            
        elif bench in ["videogen_reward_bench","videogen-reward-bench",
                       "genai_bench","genai-bench",
                       "vision_reward_db_video",]:
            res_item.update({
                "video_name":video_name,
                "prompt":prompt,
                "v_score_model":v_out, "t_score_model":t_out, "p_score_model":p_out,
                "output":raw_output
            })
        
        with open(eval_res_path,"r",encoding='utf-8') as f:
            res_data=json.load(f)
        res_data.append(res_item)
        with open(eval_res_path,"w",encoding='utf-8') as f:
            json.dump(res_data,f,indent=4,ensure_ascii=False)
        print("saved one item")


if __name__ == "__main__":    

    bench_data_dir="./bench_data"
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench",required=True,default="vs2_test_sft_17k")
    ap.add_argument("--method",required=True,default="vs2")
    ap.add_argument("--model_name_or_path",required=False)
    ap.add_argument("--bench_data_num",required=False,default='all')
    ap.add_argument("--kwargs", type=str,required=False,default="{}") 
    t_args = ap.parse_args()
    
    main(t_args)
    
