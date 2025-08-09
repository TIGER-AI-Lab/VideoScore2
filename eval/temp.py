import json

p="/data/xuan/workdir/VideoScore2/eval/res_data/res_mj_bench_video/vs2_qwen2_5vl_sft_17k_2e-4_2fps_960_720_8192_infer_2fps_old.json"
new_p="/data/xuan/workdir/VideoScore2/eval/res_data/res_mj_bench_video/vs2_qwen2_5vl_sft_17k_2e-4_2fps_960_720_8192_infer_2fps.json"
p2="/data/xuan/workdir/VideoScore2/eval/bench_data/mj_bench_video/mj_bench_video.json"

data=json.load(open(p,"r"))
data2=json.load(open(p2,"r"))

new_data=[]
for item in data:
    video_name=item["video_name"]
    for x in data2:
        if x["video_name"]==video_name:
            item["p_score_gt"]=x["consistency"]
            item["total_score"]=x["total_score"]
            new_data.append(item)

with open(new_p,"w") as f:
    json.dump(new_data,f,indent=4)