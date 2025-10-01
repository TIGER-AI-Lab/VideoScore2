# from transformers import Qwen2_5_VLForConditionalGeneration
from transformers import AutoModel, AutoTokenizer, AutoProcessor
from transformers import AutoProcessor, AutoModelForVision2Seq
from qwen_vl_utils import process_vision_info
import cv2
import os
import re

def _get_video_fps(url_or_p:str):
    cap = cv2.VideoCapture(url_or_p)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {url_or_p}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps




class eval_VideoScore2_int:
    def __init__(self,model_name):
        self.model, self.processor = self.load_model_processor(model_name)

    def load_model_processor(self,model_name):
        # with init_empty_weights():
        #     model = AutoModelForVision2Seq.from_pretrained(
        #         model_name,
        #         torch_dtype=torch.float16,
        #     )

        # device_map = infer_auto_device_map(model, max_memory={i: "40GiB" for i in range(torch.cuda.device_count())})
        model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            trust_remote_code=True,
        ).to('cuda')
        processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
            # "Qwen/Qwen2.5-VL-7B-Instruct"
        )
        return model,processor
    
    # def load_model_processor(self,model_name):
    #     # model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    #     model = AutoModelForVision2Seq.from_pretrained(
    #         model_name, torch_dtype="auto", device_map="auto"
    #     )
    #     # ).to('cuda')
        
    #     processor = AutoProcessor.from_pretrained(model_name)
    #     return model,processor        
        
    def evaluate_video(self,     
            user_prompt: str,
            video_path: str,
            kwargs: dict
        ) -> str | None:
        if not os.path.exists(video_path):
            raise ValueError(f"not exist: {video_path}")
        max_tokens=kwargs.get("max_tokens",4096)
        infer_fps=kwargs.get("infer_fps",2.0)
        if infer_fps == "raw":
            infer_fps=_get_video_fps(video_path)
        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                        "fps":infer_fps
                    },
                    {
                        "type": "text", 
                        "text": user_prompt
                    },
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        try:
            image_inputs, video_inputs = process_vision_info(messages)
        except Exception as e:
            raise ValueError(f"error when reading: {video_path}")

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            fps=infer_fps,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")
        
        generated_ids = self.model.generate(**inputs, max_new_tokens=max_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]

        pattern = r"visual quality:\s*(\d+).*?text-to-video alignment:\s*(\d+).*?physical/common-sense consistency:\s*(\d+)"
        match = re.search(pattern, output_text, re.DOTALL | re.IGNORECASE)
        if match:
            v_score_model = int(match.group(1))
            t_score_model = int(match.group(2))
            p_score_model = int(match.group(3))
        else:
            v_score_model = t_score_model = p_score_model = None
        return v_score_model, t_score_model, p_score_model, output_text
    
