import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from benchmark import VS2_QUERY_TEMPLATE, load_benchmark
import argparse

def main(args,gen_configs):
    model_name=args.model_name
    bench=args.bench
    bench_loaded_num = args.bench_loaded_num
    infer_fps=gen_configs["infer_fps"]
    if bench in ["vs2","vs2_bench","vs2-bench"]:
        bench="vs2_bench"
    elif bench_name in ["videogen_reward_bench","videogen-reward-bench"]:
        bench_name = "videogen_reward_bench"
    elif bench_name in ["mj_video_bench","mj_bench_video","mj-video-bench","mj-bench-video"]:
        bench_name = "mj_bench_video"
    elif bench_name in ["video_phy2","video_phy2_test"]:
        bench_name = "video_phy2_test"
    
    bench_data=load_benchmark(BENCH_DATA_DIR,bench,bench_loaded_num)
    
    if '/' in model_name:
        model_name=model_name.split('/')[1]
    if isinstance(infer_fps,float):
        infer_fps=int(infer_fps)
        
    eval_res_path=f"res_data/res_{bench}/{model_name}_infer_{infer_fps}fps.json"
    if "few_shot_config" in gen_configs and \
        gen_configs["few_shot_config"]["enabled"] and gen_configs["few_shot_config"]["few_shot_egs_path"] is not None:
        num_egs=gen_configs["few_shot_config"]["num_egs"]
        eval_res_path=f"res_data/res_{bench}/{model_name}_infer_{infer_fps}fps_{num_egs}shot.json"
        
    os.makedirs(os.path.dirname(eval_res_path),exist_ok=True)
    
        
    from eval_methods.open_router_api import open_router_run_one_video
    if "api_key" not in gen_configs:
        gen_configs["api_key"]=os.environ.get("OR_API_KEY","")

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
        
        print(f"Loaded existing {len(res_data)} res items for bench:{bench}, model_name:{model_name}")
        bench_data=[item for item in bench_data if item['video_name'] not in set([x['video_name'] for x in res_data])]
    
    video_paths=[os.path.abspath(f"{BENCH_DATA_DIR}/{bench}/videos/{x['video_name']}.mp4") for x in bench_data]
    user_prompts=[VS2_QUERY_TEMPLATE.substitute(t2v_prompt=x['prompt']) for x in bench_data]

    eval_outputs=[]
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:     
        futures = [executor.submit(open_router_run_one_video, user_prompt, video_path, gen_configs) for user_prompt, video_path in zip(user_prompts, video_paths)]
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
        
        if bench in ["vs2_bench","vs2-bench","vs2",
                         "video_phy2_test","videophy2_test",
                         "mj_video_bench","mj_bench_video","mj-video-bench","mj-bench-video"]:  
            if bench in ["vs2_bench","vs2-bench","vs2"]:    
                v_gt=item['visual_score']
                t_gt=item['t2v_score']
                p_gt=item['phy_score']
            elif bench in ["video_phy2_test","videophy2_test",]:
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
            
        elif bench in ["videogen_reward_bench","videogen-reward-bench",]:
            res_item={
                "video_name":video_name,
                "prompt":prompt,
                "v_score_model":v_out, "t_score_model":t_out, "p_score_model":p_out,
                "output":raw_output
            }

        if os.path.exists(eval_res_path):
            with open(eval_res_path,"r") as f:
                res_data=json.load(f)
        else:
            res_data=[]

        res_data.append(res_item)
        with open(eval_res_path,"w",encoding='utf-8') as f:
            json.dump(res_data,f,indent=4,ensure_ascii=False)
        print("saved one item")

    

if __name__ == "__main__":
    # =========================== example mllm for open-router api ===========================
    # "anthropic/claude-sonnet-4",
    # "google/gemini-2.5-pro",
    # "google/gemini-2.5-flash",
    # "openai/gpt-5",
    # "openai/gpt-5-mini",
    # "openai/o4-mini",
    # "openai/o3",
    # "x-ai/grok-4",
    # "google/gemma-3-27b-it",
    # "meta-llama/llama-4-maverick",
    # "meta-llama/llama-4-scout",
    # "thudm/glm-4.1v-9b-thinking",
    # "z-ai/glm-4.5v",
    # "qwen/qwen2.5-vl-7b-instruct",
    # "qwen/qwen2.5-vl-32b-instruct",
    # "qwen/qwen2.5-vl-72b-instruct",
    
    BENCH_DATA_DIR="bench_data"
    MAX_WORKERS=2
    few_shot_egs_path="eval_methods/utils_mllm/mllm_few_shot_egs.json"
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench",type=str,required=False,default="vs2_bench")
    ap.add_argument("--bench_loaded_num", required=False, default=10)
    ap.add_argument("--model_name",type=str, required=False,default="openai/gpt-5-mini")
    ap.add_argument("--thinking_enabled", required=False, type=bool, default=True)
    ap.add_argument("--thinking_budget", required=False, type=int, default=2048)
    ap.add_argument("--max_tokens", required=False, type=int, default=1024)
    ap.add_argument("--temperature", required=False, type=float, default=0.7)
    ap.add_argument("--infer_fps", required=False, type=float, default=2.0)
    ap.add_argument("--few_shot_enabled", required=False, type=bool, default=True)
    ap.add_argument("--few_shot_num_egs", required=False, type=int, default=3)

    args = ap.parse_args()
    gen_configs={
        "api_key":os.environ.get("OR_API_KEY",""),
        "model_name":args.model_name,
        "thinking_enabled": args.thinking_enabled,
        "thinking_budget": args.thinking_budget,
        "max_tokens":args.max_tokens,
        "temperature":args.temperature,
        "infer_fps":args.infer_fps,
        "few_shot_config":
            {
                "enabled": args.few_shot_enabled,
                "num_egs": args.few_shot_num_egs,
                "few_shot_egs_path":few_shot_egs_path,
            }          
    }
    
    main(args,gen_configs)

    
