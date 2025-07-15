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

if __name__ == "__main__":
    input_file = "thinking_cmt/sft_17k_modified.json"     
    output_folder = "thinking_split" 
    split_json_file(input_file, output_folder, chunk_size=1000)