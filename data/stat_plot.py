import json
import os
import matplotlib.pyplot as plt
import numpy as np


def prompt_sources():
    p="/data/xuan/data/videoscore2/text_prompts/all_prompts.jsonl"
    with open(p, "r", encoding="utf-8") as f:
        prompt_items = [json.loads(line) for line in f]
    
    paths=[
        f"thinking_final/{fname}" for fname in os.listdir("thinking_final") if fname.endswith('.json')
    ]
    annos=[]
    for path in paths:
        annos.extend(json.load(open(path,"r",encoding='utf-8')))