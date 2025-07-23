

import importlib
import json
import os
import requests
import argparse
from tqdm import tqdm
import time
import re

from eval_methods.vs2 import eval_VideoScore2
from benchmark import INPUT_TEMPLATE,DIM_NAMES,load_benchmark


def eval_vs2(args):
    # method=args.method
    # bench=args.bench
    # method_kwargs = json.loads(args.method_kwargs)

    method=args.get("method","vs2_test_sft_17k")
    bench=args.get("bench","sft_17k")
    bench_data_num=args.get("bench_data_num",150)
    method_kwargs = args["method_kwargs"]

    bench_data=load_benchmark(bench_data_dir,bench,bench_data_num)
    
    eval_res_path=f"res_data/res_{bench}/{method}.json"
    metrics_report_path=f"metrics_report/met_{bench}/{method}.json"
    os.makedirs(os.path.dirname(eval_res_path),exist_ok=True)
    os.makedirs(os.path.dirname(metrics_report_path),exist_ok=True)
    
    if method=="vs2":
        model_name=method_kwargs.get("model_name")
        infer_fps=method_kwargs.get("infer_fps",2.0)
        if isinstance(infer_fps,float):
            infer_fps=int(infer_fps)
        eval_res_path=f"res_data/res_{bench}/{model_name.split('/')[1]}_infer_{infer_fps}fps.json"
        metrics_report_path=f"metrics_report/met_{bench}/{model_name}.json"
        model=eval_VideoScore2(model_name)    
    
    res_data=[]
    if not os.path.exists(eval_res_path):
        with open(eval_res_path,"w",encoding='utf-8') as f:
            json.dump(res_data,f,indent=4,ensure_ascii=False)
        
    for item in bench_data:
        video_name=item['video_name']
        video_url=item['video_url']
        t2v_prompt=item['prompt']
        v_score=item['visual_score']
        t_score=item['t2v_score']
        p_score=item['phy_score']
        res_item={
            "video_name":item["video_name"],
            "video_url":video_url,
            "prompt":t2v_prompt,
            "v_score_gt":v_score,
            "t_score_gt":t_score,
            "p_score_gt":p_score,
        }
        video_local_path=f"{bench_data_dir}/{bench}/videos/{video_name}.mp4"
        
        try:
            user_prompt=INPUT_TEMPLATE.substitute(t2v_prompt=t2v_prompt)
            s_t=time.time()
            output=model.evaluate_video(user_prompt,video_local_path,method_kwargs)
            output=output[0]
            print(output[-100:])
            print(f"{v_score} {t_score} {p_score}")
            print("time cost: ",time.time()-s_t)
            
            pattern = r"visual quality:\s*(\d+).*?text-to-video alignment:\s*(\d+).*?physical/common-sense consistency:\s*(\d+)"
            match = re.search(pattern, output, re.DOTALL | re.IGNORECASE)

            if match:
                res_item["v_score_model"] = int(match.group(1))
                res_item["t_score_model"] = int(match.group(2))
                res_item["p_score_model"] = int(match.group(3))
            
            else:
                res_item["v_score_model"] = None
                res_item["t_score_model"] = None
                res_item["p_score_model"] = None
            
            res_item["output"]=output
            with open(eval_res_path,"r") as f:
                res_data=json.load(f)
            res_data.append(res_item)
            with open(eval_res_path,"w",encoding='utf-8') as f:
                json.dump(res_data,f,indent=4,ensure_ascii=False)
            print("saved one item")
        except Exception as e:
            print(e)
            print(f"error in evaluation, skipped {video_name}")
        
    
if __name__ == "__main__":
    supported_methods=[
        "vs2"
    ]
    
    supported_benchs=[
        "vs2_test_sft_17k",
    ]
    
    bench_data_dir="bench_data"
    bench_data_num=150
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name")
    ap.add_argument("--infer_fps")
    # ap.add_argument("--method_kwargs", type=str, default="{}") 
    t_args = ap.parse_args()
    model_name=t_args.model_name
    infer_fps=t_args.infer_fps
    
    if infer_fps != "raw":
        infer_fps=float(infer_fps)
        
    args={
        "method":"vs2",
        "bench":"vs2_test_sft_17k",
        "bench_data_num":bench_data_num,
        "method_kwargs":{
            # "model_name":"videoscore2/vs2_qwen2_5vl_sft_17k_1e-5_2fps_warm005_8192",
            "model_name":model_name,
            "infer_fps":infer_fps
        }
    }
    
    eval_vs2(args)
    
