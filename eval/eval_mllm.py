

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time
import re
from eval_methods.claude import claude_run_one_video
from eval_methods.gemini import gemini_run_one_video
from eval_methods.gpt import gpt_run_one_video
from eval_methods.open_router_api import open_router_run_one_video
from benchmark import INPUT_TEMPLATE, load_benchmark



def eval(args):
    # method=args.method
    # bench=args.bench
    # method_kwargs = json.loads(args.method_kwargs)
    method = args["method"]
    bench = args["bench"]
    bench_data_num = args["bench_data_num"]
    method_kwargs = args["method_kwargs"]
    model_name=args["method_kwargs"]["model_name"]
    infer_fps=args["method_kwargs"]["infer_fps"]
    
    bench_data=load_benchmark(bench_data_dir,bench,bench_data_num)
    
    if '/' in model_name:
        model_name=model_name.split('/')[1]
    if method in ["open_router","OR"]:
        model_name="open-router-"+model_name
    
    eval_res_path=f"res_data/res_{bench}/{model_name}_infer_{infer_fps}fps.json"
    metrics_report_path=f"res_metrics/met_{bench}/{model_name}.json"
    
    os.makedirs(os.path.dirname(eval_res_path),exist_ok=True)
    os.makedirs(os.path.dirname(metrics_report_path),exist_ok=True)
    
    max_workers=4
    
    if method=="claude":
        eval_one_video_func=claude_run_one_video
    
    elif method=="gpt":
        eval_one_video_func=gpt_run_one_video
        if "thinking_effort" not in method_kwargs or method_kwargs["thinking_effort"] not in ["low","medium", "high"]:
            method_kwargs["thinking_effort"] = "medium"
            
    elif method=="gemini":
        eval_one_video_func=gemini_run_one_video
    
    elif method in ["open_router","OR"]:
        eval_one_video_func=open_router_run_one_video
        if "api_key" not in method_kwargs:
            method_kwargs["api_key"]=os.environ.get(["OR_API_KEY"],"")
    
    else:
        print("model not supported")
        exit()
    
    res_data=[]
    if not os.path.exists(eval_res_path):
        with open(eval_res_path,"w",encoding='utf-8') as f:
            json.dump(res_data,f,indent=4,ensure_ascii=False)
    
    video_paths=[os.path.abspath(f"{bench_data_dir}/{bench}/videos/{x['video_name']}.mp4") for x in bench_data]
    user_prompts=[INPUT_TEMPLATE.substitute(t2v_prompt=x['prompt']) for x in bench_data]

    eval_outputs=[]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:     
        futures = [executor.submit(eval_one_video_func, item, user_prompt, video_path, eval_res_path, method_kwargs) for item, user_prompt, video_path in zip(bench_data, user_prompts, video_paths)]
        for future in tqdm(as_completed(futures), total=len(futures)):
            output = future.result()
            eval_outputs.append(output)
    

    
if __name__ == "__main__":

    bench_data_dir="bench_data"

    args={
        "method":"OR",  # "claude", "gpt", "gemini", "open_router"
        "bench":"vs2_test_sft_17k",
        "bench_data_num":150,
        "method_kwargs":{
            # "model_name":"claude-sonnet-4-20250514",
            "model_name":"anthropic/claude-sonnet-4",
            # "model_name":"google/gemini-2.5-flash",
            # "model_name":"google/gemini-2.5-pro",
            # "model_name":"google/gemma-3-27b-it",
            # "model_name":"x-ai/grok-4",
            # "model_name":"openai/gpt-4.1",
            # "model_name":"openai/o4-mini",
            # "model_name":"openai/o3",
            # "model_name":"meta-llama/llama-4-maverick",
            # "model_name":"meta-llama/llama-4-scout",
            # "model_name":"qwen/qwen2.5-vl-32b-instruct",
            # "model_name":"qwen/qwen2.5-vl-72b-instruct",
            # "model_name":"thudm/glm-4.1v-9b-thinking",
            "api_key":os.environ["OPEN_ROUTER_KEY1"],
            "thinking_enabled": True,
            "thinking_budget": 2048,
            "max_tokens":1024,
            "temperature":0.7,
            "infer_fps":4.0
        },
    }
    
    # ap = argparse.ArgumentParser()
    # ap.add_argument("--method")
    # ap.add_argument("--bench")
    # ap.add_argument("--method_kwargs", type=str, default="{}") 
    # args = ap.parse_args()

    eval(args)
    
