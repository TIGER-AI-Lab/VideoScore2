from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import cv2


def _get_video_fps(url):
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {url}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps

class eval_VideoScore2:
    def __init__(self):
        self.model = self.load_model_processor()

    def load_model_processor(self,model_name,processor_name):
        print("Loading Baseline 1 model...")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype="auto", device_map="auto"
        )
        
        processor = AutoProcessor.from_pretrained(processor_name)
        return model,processor        
        
    def evaluate(self, q_template,video_url,t2v_prompt):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_url,
                    },
                    {
                        "type": "text", 
                        "text": q_template.substitute(t2v_prompt=t2v_prompt)
                    },
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            fps=_get_video_fps(video_url),
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        )
        inputs = inputs.to("cuda")

        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text
    
