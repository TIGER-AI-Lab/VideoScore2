from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
# from models import Qwen2_5_VLForConditionalGeneration
import torch
from qwen_vl_utils import process_vision_info

# Load model and processor
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "xiaoliux/AIGVE-MACS", 
    torch_dtype=torch.bfloat16, 
    # attn_implementation="flash_attention_2"
).to("cuda:0")

processor = AutoProcessor.from_pretrained("xiaoliux/AIGVE-MACS", use_fast=True)

# Compose input message
def get_user_message(video_frames, prompt):
    messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "resized_height": 480,
                        "resized_width": 854,
                        'fps': 1,
                        "video": video_frames,
                    },
                    {
                        "type": "text",
                        "text": "You are an expert in evaluating AI-Generated Videos, you evaluate videos in the following 9 aspects: "
                                "1. technical_quality: including whether the resolution is sufficient for object recognition, whether the colors are natural, and whether there is an absence of noise or artifacts. "
                                "2. dynamic: the extent of pixel changes throughout the video, focusing on significant object or camera movements and changes in environmental factors such as daylight, weather, or seasons. "
                                "3. consistency: whether objects in the video maintain consistent properties, avoiding glitches, flickering, or unexpected changes." 
                                "4. physics: Determines if the scene adheres to physical laws." 
                                "5. element_presentence: Checks if all objects mentioned in the instructions are present in the video. "
                                "6. element_quality: Assesses the realism and fidelity of objects in the video, awarding higher scores for detailed, natural, and visually appealing appearances. "
                                "7. action_presentence: Evaluates whether all actions and interactions described in the instructions are accurately represented in the video. "
                                "8. action_quality: Measures the naturalness and smoothness of actions and interactions, with higher scores for those that are realistic, lifelike, and seamlessly integrated into the scene." 
                                "9. overall: Reflects the comprehensive quality of the video based on all metrics. "
                                "The score can be chosen from [0, 5] with whole numbers. You should also include the comment for each score. "
                                "Please output as a JSON."
                                f"The video instruction is: {prompt}"
                    },
                ],
            },
        ]
    return messages

# Example inputs
video_frames = ["/data/xuan/workdir/VideoScore2/eval/46_zhuofeng_dim1.png"]
prompt = "A tiger runs across a snowy field while snowflakes fall."

messages = get_user_message(video_frames, prompt)
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)

inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
    **video_kwargs,
).to("cuda:0")

# Inference
generated_ids = model.generate(**inputs, max_new_tokens=1500)
output_text = processor.batch_decode(
    [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)],
    skip_special_tokens=True
)[0]

print("Evaluation Result:\n", output_text)
print(type(output_text))
out_dict=eval(output_text)
print(out_dict["technical_quality"]['score'])
