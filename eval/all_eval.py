

import importlib
import json
import os
from datasets import load_dataset
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time
import re
from eval_methods.vs2 import eval_VideoScore2
from eval_methods.claude import claude_run_one_video
from eval_methods.gemini import gemini_run_one_video
from eval_methods.gpt import gpt_run_one_video
from eval_methods.open_router_api import open_router_run_one_video
from eval_methods.utils import _download_file
from benchmark import INPUT_TEMPLATE, load_benchmark



def eval():
    # method=args.method
    # bench=args.bench
    # method_kwargs = json.loads(args.method_kwargs)

    
    bench_data=load_benchmark(bench_data_dir,bench,bench_data_num)
    eval_res_path=f"res_data/res_{bench}/{method}.json"
    metrics_report_path=f"res_metrics/met_{bench}/{method}.json"
    os.makedirs(os.path.dirname(eval_res_path),exist_ok=True)
    os.makedirs(os.path.dirname(metrics_report_path),exist_ok=True)
    
    chat_configs={
            "model_name":method_kwargs.get("model_name",None),
            "thinking_enabled": method_kwargs.get("thinking_enabled", False),
            "thinking_budget": method_kwargs.get("thinking_budget", 2048),
            "max_tokens":method_kwargs.get("max_token", 1024),
            "temperature":method_kwargs.get("temperature", 0.7),
        }
    max_workers=4
    
    if method=="vs2":
        model_name=method_kwargs.get("model_name")
        eval_res_path=f"res_data/res_{bench}/{model_name.split('/')[1]}.json"
        metrics_report_path=f"metrics_report/met_{bench}/{model_name}.json"
        model=eval_VideoScore2(model_name)    
        eval_one_video_func=model.evaluate_video
        chat_configs["max_tokens"]=4096
    
    elif method=="claude":
        eval_one_video_func=claude_run_one_video
    
    elif method=="gpt":
        eval_one_video_func=gpt_run_one_video
        chat_configs["thinking_effort"]=method_kwargs.get("thinking_effort","medium")
        
    elif method=="gemini":
        eval_one_video_func=gemini_run_one_video
    
    elif method in ["open_router","OR"]:
        eval_one_video_func=open_router_run_one_video
        chat_configs["api_key"]=method_kwargs.get("api_key",os.environ.get(["OR_API_KEY"],""))
    
    else:
        print("method not supported")
        exit()
    
    res_data=[]
    if not os.path.exists(eval_res_path):
        with open(eval_res_path,"w",encoding='utf-8') as f:
            json.dump(res_data,f,indent=4,ensure_ascii=False)
    
    video_paths=[f"{bench_data_dir}/{bench}/{x['video_name']}.mp4" for x in bench_data]
    user_prompts=[INPUT_TEMPLATE.substitute(t2v_prompt=x['t2v_prompt']) for x in bench_data]

    eval_outputs=[]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:     
        futures = [executor.submit(eval_one_video_func, user_prompt, video_path, chat_configs) for user_prompt, video_path in zip(user_prompts, video_paths)]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading frames"):
            output = future.result()
            eval_outputs.append(output)
    
    assert len(bench_data)==len(eval_outputs)
    
    for x, res in zip(bench_data,eval_outputs):
        video_name=x['video_name']
        v_score=x['visual_score']
        t_score=x['t2v_score']
        p_score=x['phy_score']
        res_item={
            "video_name":x["video_name"],
            "video_url":x['video_url'],
            "prompt":x['prompt'],
            "v_score_gt":v_score,
            "t_score_gt":t_score,
            "p_score_gt":p_score,
        }

        try:
            if res is None:
                raise ValueError(f"output for {video_name} is None")
            short_res=res[-100:]
            print(short_res)
            print(f"{v_score} {t_score} {p_score}")
            
            pattern = r"visual quality:\s*(\d+).*?text-to-video alignment:\s*(\d+).*?physical/common-sense consistency:\s*(\d+)"
            match = re.search(pattern, short_res, re.DOTALL | re.IGNORECASE)

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
        
        
        # res = "{" + res.split("{")[-1].split("}")[0].strip() + "}"
        # try:
        #     eval_res = ast.literal_eval(str(res))
        #     if any(dim_name not in list(eval_res.keys()) for dim_name in DIM_NAMES):
        #         print(f"CHECK 0: key error for eval res of {video_name}")
        #         continue
        #     res_item["v_score_gt"]=v_score
        #     res_item["v_score_model"]=eval_res["visual quality"]
        #     res_item["t_score_gt"]=t_score
        #     res_item["t_score_model"]=eval_res["text-to-video alignment"]
        #     res_item["p_score_gt"]=p_score
        #     res_item["p_score_model"]=eval_res["physical/common-sense consistency"]
        #     res_data.append(res_item)
        # except Exception as e:
        #     print(e)
        #     continue
        
    # with open(eval_res_path,"w",encoding='utf-8') as f:
    #     json.dump(res_data,f,indent=4,ensure_ascii=False)
    

    
if __name__ == "__main__":
    supported_methods=[
        "vs2_sft_17k"
    ]
    
    supported_benchs=[
        "vs2_test_sft_17k",
        
    ]
    
    bench_data_dir="bench_data"
    bench = "vs2_test_sft_17k"
    bench_data_num=150
    
    method = "vs2"
    method_kwargs={
            "model_name":"DongfuJiang/vs2_qwen2_5vl_sft_17k_1e-5",
        }
    # ap = argparse.ArgumentParser()
    # ap.add_argument("--method")
    # ap.add_argument("--bench")
    # ap.add_argument("--method_kwargs", type=str, default="{}") 
    # args = ap.parse_args()
    
    eval()
    
