import os
import json
from tqdm import tqdm
from huggingface_hub import login,upload_folder, upload_file,HfApi
import shutil
import zipfile

model_code_mapping=json.load(open("const/model_code.json","r"))

code_model_mapping={v:k for k,v in model_code_mapping.items()}

endpoint_model_mapping={
    ('0000','0499'):["d","j","m","a","e","q","h","c","v","r",],
    # ('0500','0999'):["i","m","n","e","u","c","f","r","","",],
    # ('1000','1499'):["k","m","b","p","z","h","f","v","","",],
    # ('1500','1999'):["d","m","a","e","z","u","q","","","",],
    # ('2000','2499'):["j","m","n","p","q","h","","","","",],
    # ('2500','2999'):["i","b","n","z","f","","","","","",],
    # ('3000','3499'):["k","a","e","q","h","f","","","","",],
    # ('3500','3999'):["d","a","b","n","f","c","","","","",],
    # ('4000','4499'):["j","b","n","e","h","c","","","","",],
    # ('4500','4999'):["i","a","b","e","z","q","f","","","",],
}




repo_ID="hexuan21/VideoScore2_video_cache"
repo_ID2="hexuan21/VS2"


HF_TOKEN="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"

def direct_upload():
    api=HfApi()
    
    for k,v in endpoint_model_mapping.items():
        start_idx=int(k[0])
        end_idx=int(k[1])
        for code in v:
            name=code_model_mapping[code]
            
            video_dir=f"/data/xuan/videoscore2/videos/{name}"

            if any([not os.path.exists(os.path.join(video_dir,f"{i:06d}_{code}.mp4")) for i in range(start_idx,end_idx+1)]):
                continue
            
            BATCH_SIZE=100
            num_batch=int((end_idx+1-start_idx)/BATCH_SIZE)
            for j in range(num_batch):
                temp_dir=f"/data/xuan/videoscore2/temp/{start_idx}_{end_idx}/{name}_b{j}"
                for i in range(j*BATCH_SIZE,(j+1)*BATCH_SIZE):
                    video_name=f"{i:06d}_{code}.mp4"
                    src_path=os.path.join(video_dir,video_name)
                    dst_path=os.path.join(temp_dir,video_name)
                    if not os.path.exists(src_path):
                        continue
                    # if os.path.exists(dst_path):
                    #     continue
                    os.makedirs(temp_dir,exist_ok=True)
                    shutil.copy(src=src_path,dst=dst_path)
                    
                upload_folder(
                    folder_path=temp_dir,
                    path_in_repo=f"{start_idx}_{end_idx}/{name}/", 
                    repo_id=repo_ID,
                    repo_type="dataset",
                    token=HF_TOKEN,
                    run_as_future=True,
                )
                


def zip_upload():
    api=HfApi()
    
    for k,v in endpoint_model_mapping.items():
        start_idx=int(k[0])
        end_idx=int(k[1])
        for code in v:
            name=code_model_mapping[code]
            
            video_dir=f"/data/xuan/videoscore2/videos/{name}"
            if any([not os.path.exists(os.path.join(video_dir,f"{i:06d}_{code}.mp4")) for i in range(start_idx,end_idx+1)]):
                continue
            temp_dir=f"/data/xuan/videoscore2/temp/{start_idx}_{end_idx}"
            os.makedirs(temp_dir,exist_ok=True)
            
            output_zip=os.path.join(temp_dir,f"{name}.zip")
            with zipfile.ZipFile(output_zip, 'w') as zipf:
                for i in tqdm(range(start_idx,end_idx+1)):
                    video_name=f"{i:06d}_{code}.mp4"
                    src_path=os.path.join(video_dir,video_name)
                    zipf.write(src_path, video_name)                
                
            upload_file(
                path_or_fileobj=output_zip,
                path_in_repo=f"{start_idx}_{end_idx}/{name}.zip", 
                repo_id=repo_ID,
                repo_type="dataset",
                token=HF_TOKEN,
                run_as_future=True,
            )

            


def check_each_pack():
    None
    
if __name__ == "__main__":
    # direct_upload()
    zip_upload()

