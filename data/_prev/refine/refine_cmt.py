import json
import os
import argparse
from string import Template
from utils_refine_cmt import _refine_cmt_claude, _refine_cmt_gemini, _refine_cmt_gpt
from datasets import load_dataset
from tqdm import tqdm


visual_def='''
Below we will provide a segment of human evaluation comments for 'visual quality' of certain AI video, composed of keywords pointing out various issues in the video. 

The 'visual quality' cares about the video's visual and optical propertities, including 'resolution, overall clarity, local blurriness, smoothness, stability of brightness/contrast, distortion/misalignment, abrupt changes, and any other factors the affect the watching experience'. The keywords written by the annotators are also mostly derived from the above factors.
'''

t2v_def='''
Below we will provide a segment of human evaluation comments for 't2v_alignment' of certain AI video, composed of keywords pointing out various issues in the video. 

The 't2v_alignment' dimension mainly assesses whether the generated video fully and accurately depicts the elements mentioned in the text prompt, such as characters, actions, animals, etc., as well as background, quantity, color, weather, and so on. So the keywords written by annotators sometimes only indicate the elements that are missing from the video.
'''

phy_def='''
Below we will provide a segment of human evaluation comments for 'physical consistency' of certain AI video, composed of keywords pointing out various issues in the video. 

The 'physical consistency' dimension mainly examines whether there are any violations of common sense, physical laws, or any other aspects in the video that appear strange or unnatural. Most of the keywords provided by annotators point out the specific abnormalities or inconsistencies they observed in the video.
'''


refine_template=Template("""
We are collecting and processing human annotations for the quality evaluation of AI-generated videos in text-to-video generation. 
$dim_def

Please expand and polish these keywords into a **complete, natural and detailed thinking process**. And refer to the dimension quality score by a human annotator and some key frames (if provided) to make your output more reliable. 
The length of output should be 1000 to 1500 words. 

Your response must follow the format below strictly:
{
    "thinking": "<extended and refined thinking process>" (this field is only allowed to be string)
}
DO NOT include any text before or after the dict block

the quality score for this dimension (1-5 scale): 
$score
the text prompt used to generate the video: 
$prompt
anno_keywords: 
$comment                
""")



def refine_cmt(repo_id,save_path,num,model_name,model_access,append_img):

    data = load_dataset(repo_id, split="train")

    if num>=len(data):
        num=len(data)
    
    refined_comments=[]
    for i in tqdm(range(num)):
        sample=data[i]
        video_name=sample['video_name']
        video_url=sample['video_url']
        prompt=sample['prompt']
        visual_score=sample['visual_score']
        t2v_score=sample['t2v_align_score']
        phy_score=sample['phy_score']
        visual_cmt=sample['visual_comment_raw']
        t2v_cmt=sample['t2v_align_comment_raw']
        phy_cmt=sample['phy_comment_raw']
        eg_frames=sample['eg_frames']
        
        if not append_img:
            eg_frames=[]

        if "gpt" in model_name:
            visual_cmt_refined=_refine_cmt_gpt(
                model_name,model_access,visual_score,visual_cmt,prompt,eg_frames,refine_template,visual_def)
            t2v_cmt_refined=_refine_cmt_gpt(
                model_name,model_access,t2v_score,t2v_cmt,prompt,eg_frames,refine_template,t2v_def)
            phy_cmt_refined=_refine_cmt_gpt(
                model_name,model_access,phy_score,phy_cmt,prompt,eg_frames,refine_template,phy_def)
            
        elif "gemini" in model_name:
            visual_cmt_refined=_refine_cmt_gemini(
                model_name,model_access,visual_score,visual_cmt,prompt,eg_frames,refine_template,visual_def)
            t2v_cmt_refined=_refine_cmt_gemini(
                model_name,model_access,t2v_score,t2v_cmt,prompt,eg_frames,refine_template,t2v_def)
            phy_cmt_refined=_refine_cmt_gemini(
                model_name,model_access,phy_score,phy_cmt,prompt,eg_frames,refine_template,phy_def)
        
        elif "claude" in model_name:
            visual_cmt_refined=_refine_cmt_claude(
                model_name,model_access,visual_score,visual_cmt,prompt,eg_frames,refine_template,visual_def)
            t2v_cmt_refined=_refine_cmt_claude(
                model_name,model_access,t2v_score,t2v_cmt,prompt,eg_frames,refine_template,t2v_def)
            phy_cmt_refined=_refine_cmt_claude(
                model_name,model_access,phy_score,phy_cmt,prompt,eg_frames,refine_template,phy_def)
        else:
            print("model not supported, exited")
            exit()
    
        refined_comments.append({
            "video_name":video_name,
            "video_url":video_url,
            "prompt":prompt,
            "visual_score":visual_score,
            "visual_cmt_raw":visual_cmt,
            "visual_cmt_refined":visual_cmt_refined,
            "t2v_score":t2v_score,
            "t2v_cmt_raw":t2v_cmt,
            "t2v_cmt_refined":t2v_cmt_refined,
            "phy_score":phy_score,
            "phy_cmt_raw":phy_cmt,
            "phy_cmt_refined":phy_cmt_refined,
        })
        
        
    with open(save_path,"a",encoding="utf-8") as f:
        json.dump(refined_comments,f,indent=4,ensure_ascii=False)



if __name__ =="__main__":
    REPO_ID="hexuan21/VS2_raw_cmt"
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True, default='gpt-4o-mini')
    parser.add_argument('--append_img', type=int, required=True, default=1)
    parser.add_argument('--api_key', type=str, required=True,)
    parser.add_argument('--base_url', type=str, required=False, default=None)
    args = parser.parse_args()
          
    model_name=args.model_name
    model_access={
        "api_key":args.api_key,
        "base_url":args.base_url,      # only gpt series need this field
    } 
    append_img=args.append_img
    num=30
    if append_img:
        save_path=os.path.join("refined_cmt_think",f"res_{model_name}_with_img.json")
    else:
        save_path=os.path.join("refined_cmt_think",f"res_{model_name}_no_img.json")
    os.makedirs(os.path.dirname(save_path),exist_ok=True)
    refine_cmt(REPO_ID,save_path,num,
                 model_name,model_access,append_img)