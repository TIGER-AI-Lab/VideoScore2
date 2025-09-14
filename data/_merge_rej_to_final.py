import json
from datasets import load_dataset, Features, Value, Sequence, Image
import base64
import os
from PIL import Image
import io
from tqdm import tqdm


def merge_rej_to_final(resampled_p,final_p):
    with open(final_p,"r") as f:
        final_data=json.load(f)
    
    with open(resampled_p,"r") as f:
        resampled_data=json.load(f)
    
    final_data_dict={x["video_name"]:x for x in final_data}
    final_list=list(final_data_dict.keys())
    
    for item in resampled_data:
        video_name=item["video_name"]
        final_data_dict[video_name]=item

    final_data=list(final_data_dict.values())
    
    # with open(final_p,"w",encoding='utf-8') as f:
    #     json.dump(final_data,f,indent=4,ensure_ascii=False)
            