
from eval_methods.eval_vs2 import eval_VideoScore2
from template import INPUT_TEMPLATE,DIM_NAMES
import importlib
import ast
import json
import argparse


def _import_method_class(name):
    if name in ["vs2"]:
        module = importlib.import_module(f"eval_methods.eval_VideoScore2")
    return getattr(module, name) 

def load_benchmark(name):
    None

def eval(args):
    method=args.method
    benchmark=args.benchmark
    method_kwargs = json.loads(args.method_kwargs)

    eval_res_path=f"res_data/res_{benchmark}_{method}.json"
    metrics_report_path=f"metrics_report/met_{benchmark}_{method}.json"
    
    if method=="vs2":
        cls=_import_method_class(method)
        vs2_model_name=method_kwargs.get("vs2_model_name")
        vs2_processor_name=method_kwargs.get("vs2_processor_name")
        eval_res_path=f"res_data/res_{benchmark}_{vs2_model_name}.json"
        metrics_report_path=f"metrics_report/met_{benchmark}_{vs2_model_name}.json"
        model=cls(vs2_model_name,vs2_processor_name)    
        
    bench_data=load_benchmark(benchmark)
    
    res_data=[]
    for item in bench_data:
        video_name=item['video_name']
        video_url=item['video_url']
        t2v_prompt=item['prompt']
        v_score=item['v_score']
        t_score=item['t_score']
        p_score=item['p_score']
        res_item={
            "video_name":item["video_name"],
            "video_url":video_url,
            "prompt":t2v_prompt,
        }
        
        res=model.evaluate(INPUT_TEMPLATE,video_url,t2v_prompt,)
        res = "{" + res.split("{")[-1].split("}")[0].strip() + "}"
        try:
            eval_res = ast.literal_eval(str(res))
            if any(dim_name not in list(eval_res.keys()) for dim_name in DIM_NAMES):
                print(f"CHECK 0: key error for eval res of {video_name}")
                continue
            res_item["v_score_gt"]=v_score
            res_item["v_score_model"]=eval_res["visual quality"]
            res_item["t_score_gt"]=t_score
            res_item["t_score_model"]=eval_res["text-to-video alignment"]
            res_item["p_score_gt"]=p_score
            res_item["p_score_model"]=eval_res["physical/common-sense consistency"]
            res_data.append(res_item)
        except Exception as e:
            print(e)
            continue
    with open(eval_res_path,"w",encoding='utf-8') as f:
        json.dump(res_data,eval_res_path,indent=4,ensure_ascii=False)
    
    from metrics import compute_accuracy,compute_spcc,compute_plcc
    v_gt=[x["v_score_gt"] for x in res_data]
    v_pred=[x["v_score_model"] for x in res_data]
    
    t_gt=[x["t_score_gt"] for x in res_data]
    t_pred=[x["t_score_model"] for x in res_data]
    
    p_gt=[x["p_score_gt"] for x in res_data]
    p_pred=[x["p_score_model"] for x in res_data]
    
    acc_list=[compute_accuracy(v_gt,v_pred),compute_accuracy(t_gt,t_pred),compute_accuracy(p_gt,p_pred)]
    spcc_list=[compute_spcc(v_gt,v_pred),compute_spcc(t_gt,t_pred),compute_spcc(p_gt,p_pred)]
    plcc_list=[compute_plcc(v_gt,v_pred),compute_plcc(t_gt,t_pred),compute_plcc(p_gt,p_pred)]
    
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--method_kwargs", type=str, default="{}") 
    args = ap.parse_args()
    eval(args)