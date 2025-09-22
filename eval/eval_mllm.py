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
    if "few_shot_config" in method_kwargs and \
        method_kwargs["few_shot_config"]["enabled"] and method_kwargs["few_shot_config"]["few_shot_egs_path"] is not None:
        num_egs=method_kwargs["few_shot_config"]["num_egs"]
        eval_res_path=f"res_data/res_{bench}/{model_name}_infer_{infer_fps}fps_{num_egs}-shot.json"
        
    metrics_report_path=f"res_metrics/met_{bench}/{model_name}.json"
    os.makedirs(os.path.dirname(eval_res_path),exist_ok=True)
    os.makedirs(os.path.dirname(metrics_report_path),exist_ok=True)
    
    
    
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
        bench_data=[item for item in bench_data if item['video_name'] not in set([x['video_name'] for x in res_data])]
    
    video_paths=[os.path.abspath(f"{bench_data_dir}/{bench}/videos/{x['video_name']}.mp4") for x in bench_data]
    user_prompts=[VS2_QUERY_TEMPLATE.substitute(t2v_prompt=x['prompt']) for x in bench_data]

    eval_outputs=[]
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:     
        futures = [executor.submit(eval_one_video_func, user_prompt, video_path, method_kwargs) for user_prompt, video_path in zip(user_prompts, video_paths)]
        for future in tqdm(as_completed(futures), total=len(futures)):
            res_tuple = future.result()
            eval_outputs.append(res_tuple)
    
    assert len(bench_data)==len(eval_outputs),"len(bench_data)==len(eval_outputs)"

    for item,res_tuple in zip(bench_data,eval_outputs):
        video_name=item['video_name']
        prompt=item['prompt']
        if res_tuple is None:
            print(f"output for {video_name} is None")
            v_out = t_out = p_out = raw_output = None
        else:
            v_out, t_out, p_out, raw_output = res_tuple[0], res_tuple[1], res_tuple[2], res_tuple[3]
        
        if "vs2" in bench \
            or bench in ["aigve_bench","aigve-bench",
                         "video_phy","video_phy_test_public",
                         "mj_video_bench","mj_bench_video","mj-video-bench","mj-bench-video"]:  
            if "vs2" in bench:    
                v_gt=item['visual_score']
                t_gt=item['t2v_score']
                p_gt=item['phy_score']
            elif bench in ["aigve_bench","aigve-bench"]:
                v_gt=int(round((item['tech_quality']+item['ele_quality'])/2))
                t_gt=int(round((item['ele_presence']+item['act_presence'])/2))
                p_gt=item['physics']
            elif bench in ["video_phy","video_phy_test","video_phy2","video_phy2_test"]:
                v_gt=None
                t_gt=item['semantic']
                p_gt=item['physical']
            elif bench in ["mj_video_bench","mj_bench_video","mj-video-bench","mj-bench-video"]:
                v_gt=item['fineness']
                t_gt=item['alignment']
                p_gt=None
            print(f"gt: {v_gt} {t_gt} {p_gt}")  
            res_item={
                "video_name":video_name,
                "video_url":item['video_url'],
                "prompt":prompt,
                "v_score_gt":v_gt, "t_score_gt":t_gt, "p_score_gt":p_gt,
                "v_score_model":v_out, "t_score_model":t_out, "p_score_model":p_out,
                "output":raw_output
            }
            
        elif bench in ["videogen_reward_bench","videogen-reward-bench",
                       "genai_bench","genai-bench",]:
            res_item={
                "video_name":video_name,
                "prompt":prompt,
                "v_score_model":v_out, "t_score_model":t_out, "p_score_model":p_out,
                "output":raw_output
            }


        with open(eval_res_path,"r") as f:
            res_data=json.load(f)
        res_data.append(res_item)
        with open(eval_res_path,"w",encoding='utf-8') as f:
            json.dump(res_data,f,indent=4,ensure_ascii=False)
        print("saved one item")

    

if __name__ == "__main__":
    # ap = argparse.ArgumentParser()
    # ap.add_argument("--method")
    # ap.add_argument("--bench")
    # ap.add_argument("--method_kwargs", type=str, default="{}") 
    # args = ap.parse_args()
    # main(args)
    
    
    bench_data_dir="bench_data"
    MAX_WORKERS=2
    bench_data_num="all"
    few_shot_egs_path="eval_methods/utils_mllm/mllm_few_shot_egs.json"
    
    for model_name in [
        # "anthropic/claude-sonnet-4",
        # "google/gemini-2.5-pro",
        # "openai/gpt-5",
        # "openai/gpt-5-mini",
        # "meta-llama/llama-4-maverick",
        # "z-ai/glm-4.5v",
        # "qwen/qwen2.5-vl-32b-instruct",
        "qwen/qwen2.5-vl-72b-instruct",
    ]:
        
        args={
            "method":"OR",  # options: "claude", "gpt", "gemini", "open_router"
            "bench":"vs2_test_sft_27k",
            "bench_data_num":bench_data_num,
            "method_kwargs":{
                # "model_name":"anthropic/claude-sonnet-4",
                # "model_name":"google/gemini-2.5-pro",
                # "model_name":"google/gemini-2.5-flash",
                # "model_name":"openai/gpt-5",
                # "model_name":"openai/gpt-5-mini",
                # "model_name":"openai/o4-mini",
                # "model_name":"openai/o3",
                # "model_name":"x-ai/grok-4",
                # "model_name":"google/gemma-3-27b-it",
                # "model_name":"meta-llama/llama-4-maverick",
                # "model_name":"meta-llama/llama-4-scout",
                # "model_name":"thudm/glm-4.1v-9b-thinking",
                # "model_name":"z-ai/glm-4.5v",
                # "model_name":"qwen/qwen2.5-vl-7b-instruct",
                # "model_name":"qwen/qwen2.5-vl-32b-instruct",
                # "model_name":"qwen/qwen2.5-vl-72b-instruct",
                
                "model_name":model_name,
                "api_key":os.environ["OPEN_ROUTER_KEY0"],
                "thinking_enabled": True,
                "thinking_budget": 2048,
                "max_tokens":1024,
                "temperature":0.7,
                "infer_fps":2.0,
                "few_shot_config":
                    {
                        "enabled": True,
                        "num_egs": 3,
                        "few_shot_egs_path":few_shot_egs_path,
                    }          
            },
        }

        main(args)
