import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time
import re
from benchmark import VS2_QUERY_TEMPLATE, load_benchmark


def main(args):
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
    if isinstance(infer_fps,float):
        infer_fps=int(infer_fps)
        
    eval_res_path=f"res_data/res_{bench}/{model_name}_infer_{infer_fps}fps.json"
    metrics_report_path=f"res_metrics/met_{bench}/{model_name}.json"
    os.makedirs(os.path.dirname(eval_res_path),exist_ok=True)
    os.makedirs(os.path.dirname(metrics_report_path),exist_ok=True)
    
    max_workers=4
    
    if method=="claude":
        from eval_methods.claude import claude_run_one_video
        eval_one_video_func=claude_run_one_video

    elif method=="gpt":
        from eval_methods.gpt import gpt_run_one_video
        eval_one_video_func=gpt_run_one_video
        if "thinking_effort" not in method_kwargs or method_kwargs["thinking_effort"] not in ["low","medium", "high"]:
            method_kwargs["thinking_effort"] = "medium"
            
    elif method=="gemini":
        from eval_methods.gemini import gemini_run_one_video
        eval_one_video_func=gemini_run_one_video
    
    elif method in ["open_router","OR"]:
        from eval_methods.open_router_api import open_router_run_one_video
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
    user_prompts=[VS2_QUERY_TEMPLATE.substitute(t2v_prompt=x['prompt']) for x in bench_data]

    eval_outputs=[]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:     
        futures = [executor.submit(eval_one_video_func, user_prompt, video_path, method_kwargs) for user_prompt, video_path in zip(user_prompts, video_paths)]
        for future in tqdm(as_completed(futures), total=len(futures)):
            output = future.result()
            eval_outputs.append(output)
    
    assert len(bench_data)==len(eval_outputs),"len(bench_data)==len(eval_outputs)"
    
    for item,res in zip(bench_data,eval_outputs):
        video_name=item['video_name']
        prompt=item['prompt']
        if res is None:
            print(f"output for {video_name} is None")
            v_score_model = t_score_model = p_score_model = None
        
        short_res=res[-100:]
        print(short_res)
        pattern = r"visual quality:\s*(\d+).*?text-to-video alignment:\s*(\d+).*?physical/common-sense consistency:\s*(\d+)"
        match = re.search(pattern, short_res, re.DOTALL | re.IGNORECASE)
        if match:
            v_score_model = int(match.group(1))
            t_score_model = int(match.group(2))
            p_score_model = int(match.group(3))
        else:
            v_score_model = t_score_model = p_score_model = None
        
        if "vs2" in bench:
            video_name=item['video_name']
            v_score_gt=item['visual_score']
            t_score_gt=item['t2v_score']
            p_score_gt=item['phy_score']
            print(f"gt: {v_score_gt} {t_score_gt} {p_score_gt}")  
            
            res_item={
                "video_name":item["video_name"],
                "video_url":item['video_url'],
                "prompt":item['prompt'],
                "v_score_gt":v_score_gt,
                "t_score_gt":t_score_gt,
                "p_score_gt":p_score_gt,
                "v_score_model":v_score_model,
                "t_score_model":t_score_model,
                "p_score_model":p_score_model,
                "output":res
            }
            
        elif bench in ["videogen_reward_bench","videogen-reward-bench"]:
            res_item={
                "video_name":item["video_name"],
                "prompt":item['prompt'],
                "v_score_model":v_score_model,
                "t_score_model":t_score_model,
                "p_score_model":p_score_model,
                "output":output
            }
        
        elif bench in ["videogen_reward_bench","videogen-reward-bench","genai_bench","genai-bench"]:
            res_item={
                "video_name":item["video_name"],
                "prompt":item['prompt'],
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

    bench_data_dir="bench_data"

    args={
        "method":"OR",  # "claude", "gpt", "gemini", "open_router"
        "bench":"vs2_test_sft_17k",
        "bench_data_num":150,
        "method_kwargs":{
            # "model_name":"claude-sonnet-4-20250514",
            # "model_name":"anthropic/claude-sonnet-4",
            # "model_name":"google/gemini-2.5-flash",
            # "model_name":"google/gemini-2.5-pro",
            # "model_name":"google/gemma-3-27b-it",
            # "model_name":"x-ai/grok-4",
            # "model_name":"openai/gpt-4.1",
            # "model_name":"openai/o4-mini",
            # "model_name":"openai/o3",
            # "model_name":"meta-llama/llama-4-maverick",
            "model_name":"meta-llama/llama-4-scout",
            # "model_name":"thudm/glm-4.1v-9b-thinking",
            # "model_name":"qwen/qwen2.5-vl-32b-instruct",
            # "model_name":"qwen/qwen2.5-vl-72b-instruct",
            # "api_key":os.environ["OPEN_ROUTER_KEY1"],
            "api_key":"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",
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

    main(args)
    
