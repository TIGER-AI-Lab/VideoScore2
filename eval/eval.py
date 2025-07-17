
from eval_methods.vs2 import eval_VideoScore2
from template import INPUT_TEMPLATE,DIM_NAMES
import importlib
import ast
import json
import os
import requests
import argparse
from datasets import load_dataset
from tqdm import tqdm
import time
import re

def _import_method_class(name):
    if name in ["vs2"]:
        module = importlib.import_module(f"eval_methods.eval_vs2")
    return getattr(module, name) 

def _download_file(url: str, save_path: str, overwrite: bool = False, timeout: int = 15):
    chunk_size=1<<14
    if os.path.exists(save_path) and not overwrite:
        print(f"[skip] {save_path} already exists")
        return

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        bar = tqdm(total=total, unit="B", unit_scale=True, desc=os.path.basename(save_path))
        with open(save_path, "wb") as f:
            for chunk in r.iter_content(chunk_size):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
        bar.close()
    print(f"[ok] Downloaded → {save_path}")


def load_benchmark(bench_name,num=100):
    data=[]
    if bench_name == "vs2_test_sft_17k":
        repo_id="hexuan21/vs2_sft"
        url=f"https://huggingface.co/datasets/{repo_id}/resolve/main/sft_17k_test.json"
        tmp_save=f"{bench_tmp_dir}/{bench_name}/sft_17k_test.json"
        _download_file(url,tmp_save,overwrite=False)
        with open(tmp_save,"r") as f:
            data=json.load(f)
        
        data=data[:num]
        
        for x in tqdm(data):
            v_name=x["video_name"]
            v_url=x["video_url"] 
            v_save_path=f"{bench_tmp_dir}/{bench_name}/{v_name}.mp4"   
            _download_file(v_url,v_save_path)
            
    return data

def eval(args):
    # method=args.method
    # bench=args.bench
    # method_kwargs = json.loads(args.method_kwargs)

    args={
        "method":"vs2",
        "bench":"vs2_test_sft_17k",
        "method_kwargs":{
            "model_name":"DongfuJiang/vs2_qwen2_5vl_sft_17k",
        }
    }
    method=args["method"]
    bench=args["bench"]
    method_kwargs = args["method_kwargs"]
    
    bench_data=load_benchmark(bench)
    
    eval_res_path=f"res_data/res_{bench}_{method}.json"
    metrics_report_path=f"metrics_report/met_{bench}_{method}.json"
    os.makedirs(os.path.dirname(eval_res_path),exist_ok=True)
    os.makedirs(os.path.dirname(metrics_report_path),exist_ok=True)
    
    if method=="vs2":
        model_name=method_kwargs.get("model_name")
        eval_res_path=f"res_data/res_{bench}_{model_name.split("/")[1]}.json"
        metrics_report_path=f"metrics_report/met_{bench}_{model_name}.json"
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
        video_local_path=f"{bench_tmp_dir}/{bench}/{video_name}.mp4"
        
        try:
            s_t=time.time()
            output=model.evaluate_video(INPUT_TEMPLATE,video_local_path,t2v_prompt,)
            print(output[0][-100:])
            print(f"{v_score} {t_score} {p_score}")
            print("time cost: ",time.time()-s_t)
            
            # pattern = r"visual quality:\s*(\d+).*?text-to-video alignment:\s*(\d+).*?physical/common-sense consistency:\s*(\d+)"
            # match = re.search(pattern, output, re.DOTALL | re.IGNORECASE)

            # if match:
            #     res_item["v_score_model"] = int(match.group(1))
            #     res_item["t_score_model"] = int(match.group(2))
            #     res_item["p_score_model"] = int(match.group(3))
            
            # else:
            #     res_item["v_score_model"] = None
            #     res_item["t_score_model"] = None
            
            #     res_item["p_score_model"] = None
            
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
        
    with open(eval_res_path,"w",encoding='utf-8') as f:
        json.dump(res_data,eval_res_path,indent=4,ensure_ascii=False)
    
    # from metrics import compute_accuracy,compute_spcc,compute_plcc
    # v_gt=[x["v_score_gt"] for x in res_data]
    # v_pred=[x["v_score_model"] for x in res_data]
    
    # t_gt=[x["t_score_gt"] for x in res_data]
    # t_pred=[x["t_score_model"] for x in res_data]
    
    # p_gt=[x["p_score_gt"] for x in res_data]
    # p_pred=[x["p_score_model"] for x in res_data]
    
    # acc_dict={"visual":compute_accuracy(v_gt,v_pred),"t2v":compute_accuracy(t_gt,t_pred),"phy":compute_accuracy(p_gt,p_pred)}
    # spcc_dict={"visual":compute_spcc(v_gt,v_pred),"t2v":compute_spcc(t_gt,t_pred),"phy":compute_spcc(p_gt,p_pred)}
    # plcc_dict={"visual":compute_plcc(v_gt,v_pred),"t2v":compute_plcc(t_gt,t_pred),"phy":compute_plcc(p_gt,p_pred)}
    # with open(metrics_report_path,"w") as f:
    #     json.dump({
    #         "acc":acc_dict,
    #         "spcc":spcc_dict,
    #         "plcc":plcc_dict
    #     })
    
if __name__ == "__main__":
    supported_methods=[
        "vs2_sft_17k"
    ]
    
    supported_benchs=[
        "vs2_test_sft_17k",
        
    ]
    
    bench_tmp_dir="bench_temp"
    
    # ap = argparse.ArgumentParser()
    # ap.add_argument("--method")
    # ap.add_argument("--bench")
    # ap.add_argument("--method_kwargs", type=str, default="{}") 
    # args = ap.parse_args()
    
    args={}
    
    eval(args)
    
