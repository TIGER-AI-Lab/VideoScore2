import os
import json
from tqdm import tqdm
import re
import argparse
from utils_chat import _thinking_cmt_claude
from string import Template
from datasets import load_dataset, Features, Value, Sequence, Image
import warnings 
from time import sleep
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

visual_def='''
The dimension 'visual quality' cares about the video's visual and optical propertities, including 'resolution, overall clarity, local blurriness, smoothness, stability of brightness/contrast, distortion/misalignment, abrupt changes, and any other factors the affect the watching experience'. The keywords written by the annotators are also mostly derived from the above factors.
'''

t2v_def='''
The dimension 't2v_alignment' mainly assesses whether the generated video fully and accurately depicts the elements mentioned in the text prompt, such as characters, actions, animals, etc., as well as background, quantity, color, weather, and so on. So the keywords written by annotators sometimes only indicate the elements that are missing from the video.
'''

phy_def='''
The dimension 'physical consistency' mainly examines whether there are any violations of common sense, physical laws, or any other aspects in the video that appear strange or unnatural. Most of the keywords provided by annotators point out the specific abnormalities or inconsistencies they observed in the video.
'''


template=Template("""
We are collecting and processing human annotations for the quality evaluation of AI-generated videos in text-to-video generation. 

$visual_def

$t2v_def

$phy_def

With the reference of some frames of the video and the comments of 3 dimensions from a human annotator, please do your best to  give a score between 1 and 5 for these dimensions, where 1 means very bad and 5 means very good. The score should be an integer.

Your thinking process should be 1500-2500 tokens long.

Your response must follow the format below strictly:
{
    'score_visual': "<quality score for this dimension>" (this field is only allowed to be a number between 1 and 5, inclusive),
    'score_t2v': "<quality score for this dimension>" (this field is only allowed to be a number between 1 and 5, inclusive),
    'score_phy': "<quality score for this dimension>" (this field is only allowed to be a number between 1 and 5, inclusive),
}
DO NOT include any text before or after the dict block

the text prompt used to generate the video: 
$prompt

annotator comments: 
comment for 'visual quality':
$comment_visual

comment for 'text-to-video alignment':
$comment_t2v

comment for 'physical consistency':
$comment_phy
                
""")


# Thread lock for file operations
file_lock = threading.Lock()

def process_single_sample(sample, model_access, save_path):
    video_name = sample['video_name']
    prompt = sample['prompt']
    eg_frames = sample['eg_frames']
    
    refined_comment = {
        "video_name": sample['video_name'],
        "video_url": sample['video_url'],
        "prompt": sample['prompt'],
        
        "visual_score": int(sample['visual_score']),
        "visual_score_model": None,
        "visual_cmt_raw": sample['visual_comment_raw'],
        
        "t2v_score": int(sample['t2v_align_score']),
        "t2v_score_model": None,
        "t2v_cmt_raw": sample['t2v_align_comment_raw'],
        
        "phy_score": int(sample['phy_score']),
        "phy_score_model": None,
        "phy_cmt_raw": sample['phy_comment_raw'],
        
        "thinking": "",
    }
    
    if "claude" in model_access["model_name"]:
        num_try = 0
        while True:
            if num_try >= 3:
                print(f"refine comment for {video_name} failed")
                return None
            try:
                comments = [refined_comment["visual_cmt_raw"],
                           refined_comment["t2v_cmt_raw"],
                           refined_comment["phy_cmt_raw"]]

                completion = _thinking_cmt_claude(
                    model_access, comments, prompt, eg_frames, template, visual_def, t2v_def, phy_def)
                break
            except Exception as e:
                print(f"refine comment for {video_name} seems time out, sleeping for 120s")
                num_try += 1
                sleep(120)
                
        if None in completion.values():
            warnings.warn(f"thinking cmt failed for {video_name}, skipped")
            return None
        
        refined_comment["thinking"] = completion["thinking"]
        output = completion["output"]
        
        output = "{" + output.split("{")[-1].split("}")[0].strip() + "}"
        try:
            eval_res = eval(str(output))
        except:
            print(f"CHECK 0: eval failed for {video_name}, skipped")
            return None
            
        if not isinstance(eval_res, dict):
            print(f"CHECK 1: output not in correct format for {video_name}, skipped")
            return None
        if not ('score_visual' in eval_res and 'score_t2v' in eval_res and 'score_phy' in eval_res):
            print(f"CHECK 2: output dict keys are incorrect for {video_name}, skipped")
            return None
        if not (eval_res["score_visual"] in [f"{i}" for i in range(1,6)] or eval_res["score_visual"] in range(1,6)):
            print(f"CHECK 3: score_visual not in correct range for {video_name}, skipped")
            return None
        if not (eval_res["score_t2v"] in [f"{i}" for i in range(1,6)] or eval_res["score_t2v"] in range(1,6)):
            print(f"CHECK 4: score_t2v not in correct range for {video_name}, skipped")
            return None
        if not (eval_res["score_phy"] in [f"{i}" for i in range(1,6)] or eval_res["score_phy"] in range(1,6)):
            print(f"CHECK 5: score_phy not in correct range for {video_name}, skipped")
            return None
        
        visual_score_model = int(eval_res["score_visual"])
        t2v_score_model = int(eval_res["score_t2v"])
        phy_score_model = int(eval_res["score_phy"])
        
        refined_comment["visual_score_model"] = visual_score_model
        refined_comment["t2v_score_model"] = t2v_score_model
        refined_comment["phy_score_model"] = phy_score_model
        
        # Thread-safe file writing
        with file_lock:
            refined_comments = json.load(open(save_path, "r", encoding="utf-8")) if os.path.exists(save_path) else []
            refined_comments.append(refined_comment)
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(refined_comments, f, indent=4, ensure_ascii=False)
        
        return refined_comment
    else:
        print("model not supported, exited")
        return None


def thinking_cmt(repo_id, save_path, num, model_access):
    data = load_dataset(repo_id, split="train")

    if num >= len(data):
        num = len(data)
    
    # Use ThreadPoolExecutor with 20 workers
    with ThreadPoolExecutor(max_workers=20) as executor:
        # Submit all tasks
        futures = [executor.submit(process_single_sample, data[i], model_access, save_path) 
                  for i in range(num)]
        
        # Process completed tasks with progress bar
        completed_count = 0
        with tqdm(total=num, desc="Processing samples") as pbar:
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result is not None:
                        completed_count += 1
                except Exception as e:
                    print(f"Task failed with error: {e}")
                pbar.update(1)
        
        print(f"Successfully processed {completed_count} out of {num} samples")
    
    # good_items=[x for x in refined_comments 
    #             if abs(x["visual_score"]-x["visual_score_model"])<=1 and 
    #                 abs(x["t2v_score"]-x["t2v_score_model"])<=1 and
    #                 abs(x["phy_score"]-x["phy_score_model"])<=1]
    # bad_items=[x for x in refined_comments if x not in good_items]
                    
    # good_path=save_path.replace(".json","_good.json")
    # bad_path=save_path.replace(".json","_bad.json")
    # with open(good_path,"a",encoding="utf-8") as f:
    #     json.dump(good_items,f,indent=4,ensure_ascii=False)
    # with open(bad_path,"a",encoding="utf-8") as f:
    #     json.dump(bad_items,f,indent=4,ensure_ascii=False)



if __name__ =="__main__":
    REPO_ID="hexuan21/VS2_raw_cmt"
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True, default='claude-sonnet-4-20250514')
    args = parser.parse_args()
          
    # Configuration with hardcoded API key
    model_access={
        "model_name": args.model_name,
        "api_key": "",
        "base_url": None,      # only gpt series need this field
    } 
    num=50
    save_path=os.path.join("thinking_cmt",f"res_{model_access['model_name']}.json")
    os.makedirs(os.path.dirname(save_path),exist_ok=True)
    thinking_cmt(REPO_ID,save_path,num,
                 model_access)
