

import json
import os
import argparse
import time
import re

from eval_methods.vs2 import eval_VideoScore2
from benchmark import INPUT_TEMPLATE,DIM_NAMES,load_benchmark


def main(args):
    # method=args.method
    # bench=args.bench
    # method_kwargs = json.loads(args.method_kwargs)

    bench=args.get("bench","vs2_testsft_17k")
    bench_data_num=args.get("bench_data_num",150)
    kwargs = args["kwargs"]

    bench_data=load_benchmark(bench_data_dir,bench,bench_data_num)
    
    model_name=kwargs.get("model_name")
    infer_fps=kwargs.get("infer_fps",2.0)
    if isinstance(infer_fps,float):
        infer_fps=int(infer_fps)
    if '/' in model_name:
        model_name=model_name.split('/')[-1]
    eval_res_path=f"res_data/res_{bench}/{model_name}_infer_{infer_fps}fps.json"
    metrics_report_path=f"metrics_report/met_{bench}/{model_name}.json"
    os.makedirs(os.path.dirname(eval_res_path),exist_ok=True)
    os.makedirs(os.path.dirname(metrics_report_path),exist_ok=True)
    
    model=eval_VideoScore2(model_name)    
    
    res_data=[]
    if not os.path.exists(eval_res_path):
        with open(eval_res_path,"w",encoding='utf-8') as f:
            json.dump(res_data,f,indent=4,ensure_ascii=False)
        
    for item in bench_data:
        video_name=item['video_name']
        t2v_prompt=item['prompt']
        video_local_path=os.path.abspath(f"{bench_data_dir}/{bench}/videos/{video_name}.mp4")
        
        try:
            user_prompt=INPUT_TEMPLATE.substitute(t2v_prompt=t2v_prompt)
            s_t=time.time()
            output=model.evaluate_video(user_prompt,video_local_path,kwargs)
            output=output[0]
            pattern = r"visual quality:\s*(\d+).*?text-to-video alignment:\s*(\d+).*?physical/common-sense consistency:\s*(\d+)"
            match = re.search(pattern, output, re.DOTALL | re.IGNORECASE)
            if match:
                v_score_model = int(match.group(1))
                t_score_model = int(match.group(2))
                p_score_model = int(match.group(3))
            else:
                v_score_model = t_score_model = p_score_model = None
            
            print(output[-100:])
        except Exception as e:
            print(e)
            print(f"error in evaluation, skipped {video_name}")
            continue
            
        if "vs2" in bench: 
            v_score_gt=item['visual_score']
            t_score_gt=item['t2v_score']
            p_score_gt=item['phy_score']
            print(f"gt: {v_score_gt} {t_score_gt} {p_score_gt}")  
            print("time cost: ",time.time()-s_t)
            res_item={
                "video_name":video_name,
                "video_url":item['video_url'],
                "prompt":t2v_prompt,
                "v_score_gt":v_score_gt,
                "t_score_gt":t_score_gt,
                "p_score_gt":p_score_gt,
                "v_score_model":v_score_model,
                "t_score_model":t_score_model,
                "p_score_model":p_score_model,
                "output":output
            }
            with open(eval_res_path,"r") as f:
                res_data=json.load(f)
            res_data.append(res_item)
            with open(eval_res_path,"w",encoding='utf-8') as f:
                json.dump(res_data,f,indent=4,ensure_ascii=False)
            print("saved one item")
        

                
        
    
if __name__ == "__main__":    
    supported_benchs=[
        "vs2_test_sft_17k",
    ]
    
    bench_data_dir="bench_data"
    bench_data_num=150
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name")
    ap.add_argument("--infer_fps")
    # ap.add_argument("--kwargs", type=str, default="{}") 
    t_args = ap.parse_args()
    model_name=t_args.model_name
    infer_fps=t_args.infer_fps
    
    if infer_fps != "raw":
        infer_fps=float(infer_fps)
        
    args={
        "method":"vs2",
        "bench":"vs2_test_sft_17k",
        "bench_data_num":bench_data_num,
        "kwargs":{
            # "model_name":"videoscore2/vs2_qwen2_5vl_sft_17k_1e-5_2fps_warm005_8192",
            "model_name":model_name,
            "infer_fps":infer_fps
        }
    }
    
    main(args)
    
