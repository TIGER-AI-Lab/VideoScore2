import gradio as gr
import spaces
import os
import time
import json
import numpy as np
import av
import torch
from PIL import Image
import functools
from transformers import AutoProcessor, AutoConfig
from models.idefics2 import Idefics2ForSequenceClassification, Idefics2ForConditionalGeneration
from models.conversation import conv_templates
from typing import List

processor = AutoProcessor.from_pretrained("Mantis-VL/mantis-8b-idefics2-video-eval-refined-40k_4096_generation")
model = Idefics2ForConditionalGeneration.from_pretrained("Mantis-VL/mantis-8b-idefics2-video-eval-refined-40k_4096_generation", torch_dtype=torch.bfloat16).eval()

MAX_NUM_FRAMES = 24
conv_template = conv_templates["idefics_2"]

with open("./examples/all_subsets.json", 'r') as f:
    examples = json.load(f)

for item in examples:
    video_id = item['images'][0].split("_")[0]
    item['images'] = [os.path.join("./examples", video_id, x) for x in item['images']]
    item['video'] = os.path.join("./examples", item['video'])
    
with open("./examples/hd.json", 'r') as f:
    hd_examples = json.load(f)

for item in hd_examples:
    item['video'] = os.path.join("./examples", item['video'])

examples = hd_examples + examples

VIDEO_EVAL_PROMPT = """
Suppose you are an expert in judging and evaluating the quality of AI-generated videos, 
please watch the following frames of a given video and see the text prompt for generating the video, 
then give scores from 5 different dimensions:
(1) visual quality: the quality of the video in terms of clearness, resolution, brightness, and color
(2) temporal consistency, the consistency of objects or humans in video
(3) dynamic degree, the degree of dynamic changes
(4) text-to-video alignment, the alignment between the text prompt and the video content
(5) factual consistency, the consistency of the video content with the common-sense and factual knowledge
For each dimension, output a number from [1,2,3,4], 
in which '1' means 'Bad', '2' means 'Average', '3' means 'Good', 
'4' means 'Real' or 'Perfect' (the video is like a real video)
Here is an output example:
visual quality: 4
temporal consistency: 4
dynamic degree: 3
text-to-video alignment: 1
factual consistency: 2
For this video, the text prompt is "{text_prompt}",
all the frames of video are as follows: 
"""


aspect_mapping= [
    "visual quality",
    "temporal consistency",
    "dynamic degree",
    "text-to-video alignment",
    "factual consistency",
]


@spaces.GPU(duration=60)
def score(prompt:str, images:List[Image.Image]):
    if not prompt:
        raise gr.Error("Please provide a prompt")
    model.to("cuda")
    if not images:
        images = None
    
    flatten_images = []
    for x in images:
        if isinstance(x, list):
            flatten_images.extend(x)
        else:
            flatten_images.append(x)
    
    messages = [{"role": "User", "content": [{"type": "text", "text": prompt}]}]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True) 
    print(prompt)
    
    flatten_images = [Image.open(x) if isinstance(x, str) else x for x in flatten_images]
    inputs = processor(text=prompt, images=flatten_images, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    outputs = model.generate(**inputs, max_new_tokens=1024)
    generated_text = processor.decode(outputs[0, inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    return generated_text
    
def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (av.container.input.InputContainer): PyAV container.
        indices (List[int]): List of frame indices to decode.
    Returns:
        np.ndarray: np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])

def eval_video(prompt, video:str):
    container = av.open(video)

    # sample uniformly 8 frames from the video
    total_frames = container.streams.video[0].frames
    if total_frames > MAX_NUM_FRAMES:
        indices = np.arange(0, total_frames, total_frames / MAX_NUM_FRAMES).astype(int)
    else:
        indices = np.arange(total_frames)
    video_frames = read_video_pyav(container, indices)

    frames = [Image.fromarray(x) for x in video_frames]
    
    eval_prompt = VIDEO_EVAL_PROMPT.format(text_prompt=prompt)
    
    num_image_token = eval_prompt.count("<image>")
    if num_image_token < len(frames):
        eval_prompt += "<image> " * (len(frames) - num_image_token)
    
    aspect_scores = score(eval_prompt, [frames])
    return aspect_scores

def build_demo():
    with gr.Blocks() as demo:
        gr.Markdown("""
            ## Video Evaluation
            upload a video along with a text prompt when generating the video, this model will evaluate the video's quality from 7 different dimensions.
        """)
        with gr.Row():
            video = gr.Video(width=500, label="Video")
            with gr.Column():
                eval_prompt_template = gr.Textbox(VIDEO_EVAL_PROMPT.strip(' \n'), label="Evaluation Prompt Template", interactive=False, max_lines=26)
                video_prompt = gr.Textbox(label="Text Prompt", lines=1)
                with gr.Row():
                    eval_button = gr.Button("Evaluate Video")
                    clear_button = gr.ClearButton([video, video_prompt])
                eval_result = gr.Textbox(label="Evaluation result", interactive=False, lines=7)
                # eval_result = gr.Json(label="Evaluation result")
                
        
        eval_button.click(
            eval_video, [video_prompt, video], [eval_result]
        )
        
        dummy_id = gr.Textbox("id", label="id", visible=False, min_width=50)
        dummy_output = gr.Textbox("reference score", label="reference scores", visible=False, lines=7)
        
        gr.Examples(
            examples=
            [
                [
                    item['id'], 
                    item['prompt'],
                    item['video'],
                    item['conversations'][1]['value']
                ] for item in examples
            ],
            inputs=[dummy_id, video_prompt, video, dummy_output],
        )        
        
    return demo    
    

if __name__ == "__main__":
    demo = build_demo()
    demo.launch(share=True)