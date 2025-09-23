import json
from datasets import load_dataset, Features, Value, Sequence, Image
import base64
import os
from PIL import Image
import io
from tqdm import tqdm

def get_base64_str():
    def _base64_str(pil):
        buffered = io.BytesIO()
        img = pil.convert("RGB")  
        img.save(buffered, format="JPEG")  
        base64_str = base64.standard_b64encode(buffered.getvalue()).decode("utf-8")
        return base64_str

    REPO_ID="hexuan21/VS2_raw_cmt"
    num=50

    data = load_dataset(REPO_ID, split="train")

    new_data=[]

    for i in range(num):
        sample=data[i]
        eg_frames=sample['eg_frames']
        new_data.append({
            "video_name": sample['video_name'],
            "prompt": sample['prompt'],
            "eg_frames_base64": [_base64_str(frame) for frame in eg_frames],
        })
        
    json.dump(new_data, open("examples_base64.json", "w", encoding="utf-8"), indent=4, ensure_ascii=True)

def build_few_shot():
    data2=json.load(open("few_shot_examples.json", "r", encoding="utf-8"))

    data=json.load(open("examples_base64.json", "r", encoding="utf-8"))

    for x in data:
        for i,y in enumerate(data2):
            video_name=x['video_name']
            if x['video_name']==y["video_name"]:
                data2[i]['frame_base64_list']=x['eg_frames_base64']
                print(f"FIND, for {video_name}")
                print(len(data2[i]['frame_base64_list']))
                print(len(data2[i]['frame_base64_list'][0]))
                break

    json.dump(data2, open("few_shot_examples_new.json", "w", encoding="utf-8"), indent=4, ensure_ascii=True)    


def check_hf_files():
    from huggingface_hub import list_repo_files
    repo_id = "hexuan21/vs2_raw_comment"
    files = list_repo_files(repo_id=repo_id, repo_type="dataset")
    anno_paths=[
            f"raw_anno/com_5k.json",
            f"raw_anno/1.json",
            f"raw_anno/2.json",
            f"raw_anno/3.json",
            f"raw_anno/4.json",
            f"raw_anno/5.json",
            f"raw_anno/13.json",
            f"raw_anno/14.json",
            f"raw_anno/15.json",
            f"raw_anno/17.json",
            f"raw_anno/18.json",
            f"raw_anno/19.json",
            f"raw_anno/20.json",
            f"raw_anno/21.json",
            f"raw_anno/22.json",
            f"raw_anno/23.json",
            f"raw_anno/24.json",
            f"raw_anno/29.json",
            f"raw_anno/30.json",
            f"raw_anno/31.json",
            f"raw_anno/32.json",
            f"raw_anno/53.json",
            f"raw_anno/54.json",
            f"raw_anno/55.json",
            f"raw_anno/61.json",
            f"raw_anno/69.json",
            f"raw_anno/70.json"
    ]

    fs=[f"{x.split('/')[1].split('.')[0]}.parquet" for x in anno_paths]
    for f in fs:
        target_file = f
        if target_file in files:
            print("✅ Found:", target_file)
        else:
            print("❌ Not found:", target_file)




def split_batchs():

    dict1={
    9:
    "67ff8a3c97cbd9edfc8fbcee",
    10:
    "67ff8a3c97cbd9edfc8fbee3",

    }


    path="VideoScore2.json"
    data=json.load(open(path,"r",encoding='utf-8'))
    for batch_name, uid in dict1.items():
        new_data=[]
        for x in data:
            if str(x["batchId"]) == uid:
                new_data.append(x)
        print(f"{batch_name}, {len(new_data)}")
        if len(new_data)==0:
            continue
        with open(f"anno_raw/{batch_name}.json","w",encoding='utf-8') as f:
            json.dump(new_data,f,indent=4,ensure_ascii=False)



def split_json_file(input_path, output_dir, chunk_size=1000):
    import json
    import os
    os.makedirs(output_dir, exist_ok=True)

    # 读取 JSON 数据
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    total = len(data)
    num_chunks = (total + chunk_size - 1) // chunk_size
    print(f"Total items: {total}, Splitting into {num_chunks} chunks...")

    for i in range(num_chunks):
        chunk = data[i * chunk_size : (i + 1) * chunk_size]
        out_path = os.path.join(output_dir, f"thinking_17k_{i}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(chunk, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(chunk)} items → {out_path}")


def prelabel(p,new_p):
    EXCL_MODELS=['stepvideo_t2v','stepvideo_t2v_low_vram','kling','sora','wanx21_14b','ruyi',]

    V_4_MODELS=['lavie_base','magictime','cogvideox_5b','wanx21_1_3b','videocrafter2','opensora_plan_v1_3','pika_v2_2','cogvideox15_5b',]
    V_3_MODELS=['anidiff','cogvideox_2b','ltx_video_095','mochi1_preview','opensora_v1_2','latte',]
    V_2_MODELS=['vchitect2','hotshot_xl',]
    V_1_MODELS=['ltx_video_091','text2video_zero','modelscope','zeroscope']

    P_4_MODELS=['magictime','wanx21_1_3b','mochi1_preview','opensora_plan_v1_3','pika_v2_2','cogvideox15_5b',]
    P_3_MODELS=['ltx_video_095','cogvideox_5b','lavie_base','hotshot_xl','videocrafter2','latte',]
    P_2_MODELS=['opensora_v1_2','cogvideox_2b','anidiff','vchitect2',]
    P_1_MODELS=['ltx_video_091','vchitect2','text2video_zero','modelscope','zeroscope']
    
    SCORE={
        "5":"5-Very Good",
        "4":"4-Very Good",
        "3":"3-Medium",
        "2":"2-Very Poor",
        "1":"1-Very Poor",
    }
    
    PRE_LABELS=[
            {
                "id": 1,
                "hash": "1_视觉质量评分",
                "label": "1_视觉质量评分",
                "value": "",
                "drawType": "QUESTION",
                "count": 1
            },
            {
                "id": 2,
                "hash": "1_视觉质量描述",
                "label": "1_视觉质量描述",
                "value": "",
                "drawType": "QUESTION",
                "count": 1
            },
            {
                "id": 3,
                "hash": "1_文本符合度评分",
                "label": "1_文本符合度评分",
                "value": "",
                "drawType": "QUESTION",
                "count": 1
            },
            {
                "id": 4,
                "hash": "1_文本符合度描述",
                "label": "1_文本符合度描述",
                "value": "",
                "drawType": "QUESTION",
                "count": 1
            },
            {
                "id": 5,
                "hash": "1_物理符合度评分",
                "label": "1_物理符合度评分",
                "value": "",
                "drawType": "QUESTION",
                "count": 1
            },
            {
                "id": 6,
                "hash": "1_物理符合度描述",
                "label": "1_物理符合度描述",
                "value": "",
                "drawType": "QUESTION",
                "count": 1
            }
        ]
    
    with open(p,'r') as f:
        data=json.load(f)
    
    for x,idx in enumerate(data):
        url=x["info"]["data"][2]["content"]
        video_name = url.split("/")[-1].split(".")[0]
        t2v_model=url.split('/')[-2]
        pre_labels=PRE_LABELS
        if t2v_model in EXCL_MODELS:
            pre_labels[0]["value"]=SCORE["5"]
            pre_labels[2]["value"]=SCORE["5"]
            pre_labels[4]["value"]=SCORE["5"]
        else:
            pre_labels[2]["value"]=SCORE["3"]
            if t2v_model in V_4_MODELS:
                pre_labels[0]["value"]=SCORE["4"]
            if t2v_model in V_3_MODELS:
                pre_labels[0]["value"]=SCORE["3"]
            if t2v_model in V_2_MODELS:
                pre_labels[0]["value"]=SCORE["2"]
            if t2v_model in V_1_MODELS:
                pre_labels[0]["value"]=SCORE["1"]
                
            if t2v_model in P_4_MODELS:
                pre_labels[4]["value"]=SCORE["4"]
            if t2v_model in P_3_MODELS:
                pre_labels[4]["value"]=SCORE["3"]
            if t2v_model in P_2_MODELS:
                pre_labels[4]["value"]=SCORE["2"]
            if t2v_model in P_1_MODELS:
                pre_labels[4]["value"]=SCORE["1"]
        data[idx]["preData"]=pre_labels
    
    with open(new_p,"w",encoding='utf-8') as f:
        json.dump(data,f,indent=4,ensure_ascii=False)

def _get_video_fps(url_or_p:str):
    import cv2
    cap = cv2.VideoCapture(url_or_p)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {url_or_p}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps


def collect_video():
    import zipfile
    from utils_fetch_f_v import _fetch_video_single
    from concurrent.futures import ThreadPoolExecutor, as_completed
    batch_name="63"
    anno_path=f"temp/{batch_name}.json"
    annos=json.load(open(anno_path,"r",encoding='utf-8'))
    video_paths=[]
    video_names=[]
    video_urls=[]
    txt_list=[]
    f_v_save_dir="/data/xuan/videoscore2/f_v_all"
    for anno in annos:
        prompt_en = (
            anno["info"]["data"][1]["content"]
            .split("English Prompt", 1)[1]
            .split("\n", 1)[0]
            .strip(". :\n")
        )
        
        prompt_cn=anno["info"]["data"][1]["content"].split("翻译为中文的Prompt", 1)[1].strip(". :\n")
        url = anno["info"]["data"][2]["content"]
        video_name = url.split("/")[-1].split(".")[0]
        video_urls.append(url)
        video_names.append(video_name)
        t2v_model = url.split("/")[-2]
        video_paths.append(f"{f_v_save_dir}/videos/{video_name}.mp4")
        txt_list.append(f"{video_name}\n{prompt_cn}        {prompt_en}")
    
    max_workers=8
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_fetch_video_single, name, url, f_v_save_dir) for name, url in zip(video_names,video_urls)]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading videos"):
            result = future.result()
            if result is None:
                continue  
    
    video_zip=f"temp/{batch_name}_videos.zip"
    print("zipping videos...")
    with zipfile.ZipFile(video_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for v_p in tqdm(video_paths):
            f_name=v_p.split('/')[-1]
            if v_p.endswith('.mp4'):
                zipf.write(v_p, arcname=f_name)     

    with open(f"temp/{batch_name}.txt","w") as f:
        for item in txt_list:
            f.write(f"{item}\n\n")


def merge_rej_to_final():
    batch_names=[
        1,2,3,4, 
        5,9,
        13,14,15,16,
        17,18,19,20,74, 
        21,22,23,24,  
        29,30,31,32,75,
        33,34,85,86,
        38,
        81,82,83,  
        45,46,47,48,
        53,54,55,56,
        
        61,62,78,79,
        69,70,71,80,
    
        "com_5k_0",
        "com_5k_1",    
        "com_5k_2",    
        "com_5k_3",   
        "com_5k_4", 
    ]
    
    for batch_name in batch_names:
        p1=f"thinking_final/final_resample_rej/rej_{batch_name}.json"
        p2=f"thinking_final/final_{batch_name}.json"
        with open(p1,"r") as f:
            data1=json.load(f)
        with open(p2,"r") as f:
            data2=json.load(f)
        data2.extend(data1)
    
        with open(p2,"w",encoding='utf-8') as f:
            json.dump(data2,f,indent=4,ensure_ascii=False)

    

def aside_prompts():
    p="/data/xuan/data/videoscore2/text_prompts/all_prompts.jsonl"
    with open(p, "r", encoding="utf-8") as f:
        prompt_items = [json.loads(line) for line in f]
    
    paths=[
        f"thinking_final/{fname}" for fname in os.listdir("thinking_final") if fname.endswith('.json')
    ]
    used_annos=[]
    for path in paths:
        used_annos.extend(json.load(open(path,"r",encoding='utf-8')))
    used_video_names=[x['video_name'].split("_")[0] for x in used_annos]
    used_video_names=list(set(used_video_names))
    
    aside_prompts_items=[x for x in prompt_items if x['video_id'] not in used_video_names]
    aside_prompts=[x['text'] for x in aside_prompts_items]
    
    with open("vs2_aside_propmts.json","w") as f:
        json.dump(aside_prompts_items,f,indent=4)
    
    with open("vs2_aside_prompts.txt", "w", encoding="utf-8") as f:
        for item in aside_prompts:
            f.write(item + "\n")
    

if __name__ == "__main__":
    # input_file = "thinking_cmt/sft_17k_modified.json"     
    # output_folder = "thinking_split" 
    # split_json_file(input_file, output_folder, chunk_size=1000)
    
    # p="/home/brantley/workdir/VideoScore2/data/0000_0499_videoscore_upload.json"
    # with open(p,"r") as f:
    #     ds=json.load(f)
    # with open(p,"w",encoding='utf-8') as f:
    #     json.dump(ds,f,indent=4,ensure_ascii=False) 
        
    # data = load_dataset("hexuan21/vs2_raw_comment", data_files=f"no_comment_5_trial.parquet",split="train")
    
    # print(data[0]["visual_comment_raw"])
    

    # dir="/data/xuan/workdir/VideoScore2/data/thinking_final/final_resample_rej"
    # dir="/data/xuan/workdir/VideoScore2/data/thinking_final"
    # data=[]
    # f_num=0
    # for f in os.listdir(dir):
    #     if f.endswith(".json"):
    #         f_num+=1
    #         p=os.path.join(dir,f)
    #         with open(p,"r") as f:
    #             ds=json.load(f)
    #         data.extend(ds)
    # print(f_num)   
    # print(len(data))
    
    
    # aside_prompts()
    
    
    
    None

            

    
    


            