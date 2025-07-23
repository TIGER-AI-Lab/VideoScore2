import json
import os
from tqdm import tqdm
from benchmark import load_benchmark
import time

def main(args):
    bench = args["bench"]
    bench_data_num = args["bench_data_num"]
    method_name = args["feat_method_name"]
    method_kwargs = args.get("metric_kwargs", {})

    bench_data = load_benchmark(bench_data_dir, bench, bench_data_num)
    eval_res_path = f"res_data_feat/{bench}/{method_name}.json"
    metrics_report_path=f"metrics_report/met_{bench}/{method_name}.json"
    os.makedirs(os.path.dirname(eval_res_path),exist_ok=True)
    os.makedirs(os.path.dirname(metrics_report_path),exist_ok=True)

    
    if method_name.lower() == "brisque":
        from eval_methods.feature_based.brisque import brisque_output
        feat_method_func = brisque_output
        model_or_process=None 
        
    elif method_name == "piqe":
        from eval_methods.feature_based.piqe import piqe_output
        feat_method_func = piqe_output
        model_or_process=None
        
    elif method_name in ["ssim-sim","ssim_sim","ssim"]:
        from eval_methods.feature_based.ssim_sim import ssim_sim_output
        feat_method_func = ssim_sim_output
        model_or_process=None
        
    elif method_name.lower() in ["clip-sim","clip_sim"]:
        from eval_methods.feature_based.clip_sim import clip_sim_output
        from transformers import CLIPProcessor, CLIPModel
        import torch
        feat_method_func = clip_sim_output
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(device)
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
        model_or_process = [model,processor]
        
    elif method_name.lower() in ["dino-sim","dino_sim","dino"]:
        from eval_methods.feature_based.dino_sim import compute_dino_similarity
        from torchvision.models import vit_b_16
        model_or_process = vit_b_16(pretrained=True).to("cuda")
        model_or_process.eval()
        feat_method_func = compute_dino_similarity
    
    elif method_name.lower() in ["clip-score","clip_score"]:
        from eval_methods.feature_based.clip_score import clip_score_output
        from transformers import CLIPProcessor, CLIPModel
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_name = "openai/clip-vit-base-patch32"
        model = CLIPModel.from_pretrained(model_name).to(device)
        processor = CLIPProcessor.from_pretrained(model_name)
        model_or_process = [model,processor,device]
        feat_method_func = clip_score_output
        
    elif method_name.lower() in ["xclip-score","xclip_score","xclip"]:
        from eval_methods.feature_based.x_clip_score import x_clip_score_output
        from transformers import AutoTokenizer, AutoModel, AutoProcessor
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_name = "microsoft/xclip-base-patch32"
        model = AutoModel.from_pretrained(model_name).to(device)
        processor = AutoProcessor.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model_or_process = [model,processor,tokenizer,device]
        feat_method_func = x_clip_score_output
        
    else:
        raise ValueError(f"Unsupported metric: {method_name}")

    res_data=[]
    if not os.path.exists(eval_res_path):
        with open(eval_res_path,"w",encoding='utf-8') as f:
            json.dump(res_data,f,indent=4,ensure_ascii=False)
            
    for item in tqdm(bench_data):
        video_name=item['video_name']
        t2v_prompt=item['prompt']
        video_path = os.path.abspath(f"{bench_data_dir}/{bench}/videos/{video_name}.mp4")
        try:
            s_t=time.time()
            score = feat_method_func(model_or_process,video_path, t2v_prompt)
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
                "v_score_model":score,
                "t_score_model":score,
                "p_score_model":score,
                "output":""
            }

            with open(eval_res_path,"r") as f:
                res_data=json.load(f)
            res_data.append(res_item)
            with open(eval_res_path,"w",encoding='utf-8') as f:
                json.dump(res_data,f,indent=4,ensure_ascii=False)
            print("saved one item")


if __name__ == "__main__":
    bench_data_dir = "bench_data"
    
    args = {
        "bench": "vs2_test_sft_17k",
        "bench_data_num": 150,
        "feat_method_name": "DINO-sim",
        "metric_kwargs": {
        }
    }
    main(args)
