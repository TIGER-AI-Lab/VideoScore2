import json
import os
import shutil
from tqdm import tqdm


def copy_video_anno_try():
    t2v_models=["anidiff","gen2","latte","lavie_base","lvdm","magictime","morph_studio","open_sora_v_1_2","open_sora_v_1_1","vc2"]
    model_codes=['a','b','c','d','e','f','g','h','i','j']

    root_dir="/data/xuan/video_eval/leader_bd/videos"
    dest_dir="/data/xuan/videoscore2/anno_page/videoscore2_anno/videos_display"

    start=50
    num_video=50

    for model,model_code in zip(t2v_models,model_codes):
        video_dir=f"{root_dir}/{model}"
        for video in tqdm(sorted(os.listdir(video_dir))[start:start+num_video]):
            dest_idx=int(video.split("_")[1])
            dest_path=f"{dest_dir}/{dest_idx:06d}_{model_code}.mp4"
            shutil.copy(f"{video_dir}/{video}",dest_path)
        
    print(sorted(os.listdir(dest_dir))[:10])

    L=[]
    for i in range(start,start+num_video):
        for model_code in model_codes:
            L.append(f"{i:06d}_{model_code}") 
    
    with open("/data/xuan/videoscore2/anno_page/text_files/sampled_id.json","w") as f:
        json.dump({"video_ids":L},f,indent=4)



# import hashlib 
# import ast
# import shutil

# prompt_path="/home/brantley/workdir/VideoScore2/prepare_video/prompts/prompt_sora.jsonl"
# all_items=[json.loads(x) for x in open(prompt_path,"r")]
# all_texts=[x['text'] for x in all_items]

# t="/data/xuan/videoscore2/videos/sora_raw/result.txt"
# with open(t, 'r', encoding='utf-8') as f:
#     lines = [line.strip() for line in f]

# print(len(os.listdir("/data/xuan/videoscore2/videos/sora_raw/video")))

# for x in tqdm(lines):
#     x=ast.literal_eval(x)
#     prompt=x['prompt']
#     id=""
#     if prompt not in all_texts:
#         pass
#         # if len(x["url"])==0:
#         #     # print(x)
#         #     continue
#         # md5 = hashlib.md5()
#         # md5.update(x['url'].encode('utf-8'))
#         # fname = md5.hexdigest() + '.mp4'
#         # shutil.move(f"/data/xuan/videoscore2/videos/sora_raw/video/{fname}",f"/data/xuan/videoscore2/videos/sora_bad/{fname}.mp4")
#     else:
#         for y in all_items:
#             if prompt==y['text']:
#                 id=y['video_id']
#         if len(x["url"])==0:
#             # print(x)
#             continue
        
#         md5 = hashlib.md5()
#         md5.update(x['url'].encode('utf-8'))
#         fname = md5.hexdigest() + '.mp4'
#         if not os.path.exists(f"/data/xuan/videoscore2/videos/sora_raw/video/{fname}"):
#             print(fname)
#             continue
#         os.makedirs(f"/data/xuan/videoscore2/videos/sora",exist_ok=True)
#         shutil.move(f"/data/xuan/videoscore2/videos/sora_raw/video/{fname}",f"/data/xuan/videoscore2/videos/sora/{id}_s.mp4")


# print(len(os.listdir("/data/xuan/videoscore2/videos/sora")))
# print(len(os.listdir("/data/xuan/videoscore2/videos/sora_bad")))

# for i in range(1000,1500):
#     if not os.path.exists(f"/data/xuan/videoscore2/videos/sora/{i:06d}_s.mp4"):
#         print(i)



