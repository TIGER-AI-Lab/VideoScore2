

import json
import os
import argparse
import time
import re
from tqdm import tqdm
from benchmark import VS2_QUERY_TEMPLATE,VS1_REG_QUERY_TEMPLATE,AIGVE_MACS_QUERY_TEMPLATE,DIM_NAMES,load_benchmark
from string import Template

def main(args):
    # method=args.method
    # bench=args.bench
    # method_kwargs = json.loads(args.method_kwargs)

    method=args.get("method","vs2")
    bench=args.get("bench","vs2_testsft_17k")
    bench_data_num=args.get("bench_data_num",150)
    kwargs = args["kwargs"]

    bench_data=load_benchmark(bench_data_dir,bench,bench_data_num)
    print("benchmark data loaded.")
    
    if method.lower() == "vs2":
        from eval_methods.vs2 import eval_VideoScore2
        model_name_or_path=kwargs.get("model_name_or_path")
        model=eval_VideoScore2(model_name_or_path)    
        q_template=VS2_QUERY_TEMPLATE
        
    elif method.lower() == "vs1":
        from eval_methods.vs1 import eval_VideoScore1
        model_name_or_path=kwargs.get("model_name_or_path")
        model=eval_VideoScore1(model_name_or_path) 
        q_template=VS1_REG_QUERY_TEMPLATE
    
    elif method.lower() == "aigve_macs":
        from eval_methods.aigve_macs import eval_AIGVE_MACS
        model_name_or_path=kwargs.get("model_name_or_path")
        model=eval_AIGVE_MACS(model_name_or_path) 
        q_template=AIGVE_MACS_QUERY_TEMPLATE
    
    elif method.lower() == "video_reward":
        from eval_methods.video_reward import eval_VideoReward
        model_name_or_path=kwargs.get("model_name_or_path")
        model=eval_VideoReward(model_name_or_path) 
        q_template=Template("""$t2v_prompt""")
    
    elif method.lower() == "vision_reward":
        from eval_methods.vision_reward import eval_VisionReward
        model_name_or_path=kwargs.get("model_name_or_path")
        model=eval_VisionReward(model_name_or_path) 
        q_template=Template("""$t2v_prompt""")
    
    elif method.lower() == "video_phy2_auto_eval":
        from eval_methods.video_phy2 import eval_VideoPhy2
        model_name_or_path=kwargs.get("model_name_or_path")
        model=eval_VideoPhy2(model_name_or_path) 
        q_template=Template("""$t2v_prompt""")
        
    # elif method.lower() in ["mj","mj_video"]:
    #     from eval_methods.mj_video import InternVL2VideoEvaluator
    #     model_name=kwargs.get("model_name")
    #     model=InternVL2VideoEvaluator(model_name)
    #     print("model loaded.")
    #     q_template=VS2_QUERY_TEMPLATE
        
    #     if '/' in model_name:
    #         model_name=model_name.split('/')[-1]
    #     eval_res_path=f"res_data/res_{bench}/{model_name}.json"
        
    else:
        print("model not supported")
        exit()
        
    if '/' in model_name_or_path:
        model_name_or_path=model_name_or_path.split('/')[-1]
    eval_res_path=f"res_data/res_{bench}/{model_name_or_path}.json"
    if "vs2" in method:
        infer_fps=kwargs.get("infer_fps",2.0)
        if isinstance(infer_fps,float):
            infer_fps=int(infer_fps)
        eval_res_path=f"res_data/res_{bench}/{model_name_or_path}_infer_{infer_fps}fps.json"
    
    metrics_report_path=f"metrics_report/met_{bench}/{model_name_or_path}.json"
    os.makedirs(os.path.dirname(eval_res_path),exist_ok=True)
    os.makedirs(os.path.dirname(metrics_report_path),exist_ok=True)
    res_data=[]
    if not os.path.exists(eval_res_path):
        with open(eval_res_path,"w",encoding='utf-8') as f:
            json.dump(res_data,f,indent=4,ensure_ascii=False)
        
    for item in tqdm(bench_data):
        video_name=item['video_name']
        prompt=item['prompt']
        path_or_url=os.path.abspath(f"{bench_data_dir}/{bench}/videos/{video_name}.mp4")
        
        try:
            user_prompt=q_template.substitute(t2v_prompt=prompt)
            s_t=time.time()
            v_out, t_out, p_out, raw_output = model.evaluate_video(user_prompt, path_or_url, kwargs)
            print("time cost: ",time.time()-s_t)
            
        except Exception as e:
            print(f"{e}\nerror in evaluation, skipped {video_name}")
            continue
           
        if "vs2" in bench \
            or bench in ["aigve_bench","aigve-bench",
                         "video_phy","video_phy_test_public",
                         "mj_video_bench","mj_bench_video","mj-video-bench","mj-bench-video"]:  
            if "vs2" in bench:    
                v_gt=item['visual_score']
                t_gt=item['t2v_score']
                p_gt=item['phy_score']
            elif bench in ["aigve_bench","aigve-bench"]:
                v_gt=int(round((item['technical_quality']+item['element_quality']+item['action_quality'])/3))
                t_gt=int(round((item['element_presence']+item['action_presence'])/2))
                p_gt=item['physics']
            elif bench in ["video_phy","video_phy_test","video_phy2","video_phy2_test"]:
                v_gt=None
                t_gt=item['semantic']
                p_gt=item['physical']
            elif bench in ["mj_video_bench","mj_bench_video","mj-video-bench","mj-bench-video"]:
                v_gt=item['fineness']
                t_gt=item['alignment']
                p_gt=None
            elif bench in ["tvge"]:
                v_gt=item['video_quality_score']
                t_gt=item['text_alignment_score']
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

    
    bench_data_dir="bench_data"
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench",required=True,default="vs2_test_sft_17k")
    ap.add_argument("--method",required=True,default="vs2")
    ap.add_argument("--model_name_or_path",required=True)
    ap.add_argument("--bench_data_num",required=False,default=150)
    ap.add_argument("--infer_fps",required=False,default=2.0)
    ap.add_argument("--kwargs", type=str,required=False,default="{}") 
    t_args = ap.parse_args()
    bench=t_args.bench
    method=t_args.method
    model_name_or_path=t_args.model_name_or_path
    bench_data_num=t_args.bench_data_num
    infer_fps=t_args.infer_fps
    
    if infer_fps != "raw":
        infer_fps=float(infer_fps)
        
    args={
        "method":method,
        "bench":bench,
        "bench_data_num":bench_data_num,
        "kwargs":{
            # "model_name":"videoscore2/vs2_qwen2_5vl_sft_17k_1e-5_2fps_warm005_8192",
            "model_name_or_path":model_name_or_path,
            "infer_fps":infer_fps
        }
    }
    
    main(args)
    
