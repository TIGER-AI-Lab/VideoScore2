from transformers import AutoProcessor, AutoModelForVision2Seq, AutoTokenizer
from qwen_vl_utils import process_vision_info
import torch
import numpy as np
import cv2, os, re

def _get_video_fps(url_or_p:str):
    cap = cv2.VideoCapture(url_or_p)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {url_or_p}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps

class eval_VideoScore2_float:
    def __init__(self, model_name: str):
        self.model, self.processor = self.load_model_processor(model_name)

        self.tokenizer = getattr(self.processor, "tokenizer", None)
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                use_fast=False,
            )

    def load_model_processor(self, model_name):
        model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            trust_remote_code=True,
        ).to("cuda")
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        return model, processor
    
    
    def evaluate_video(self,     
            user_prompt: str,
            video_path: str,
            kwargs: dict
        ) -> str | None:
        if not os.path.exists(video_path):
            raise ValueError(f"not exist: {video_path}")
        max_tokens=kwargs.get("max_tokens",4096)
        infer_fps=kwargs.get("infer_fps",2.0)
        temperature=kwargs.get("temperature",0.7)
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
                        "text": user_prompt,
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
        
        gen_out = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            output_scores=True,
            return_dict_in_generate=True,
            do_sample=True,
            temperature=temperature,
        )
        sequences = gen_out.sequences  
        scores = gen_out.scores        

        input_len = inputs["input_ids"].shape[1]
        
        gen_token_ids = sequences[0, input_len:].tolist()
            
        output_text = self.processor.batch_decode(
            sequences[:, input_len:], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        pattern = r"visual quality:\s*(\d+).*?text-to-video alignment:\s*(\d+).*?physical/common-sense consistency:\s*(\d+)"
        match = re.search(pattern, output_text, re.DOTALL | re.IGNORECASE)
        if match:
            v_score_model = int(match.group(1))
            t_score_model = int(match.group(2))
            p_score_model = int(match.group(3))
        else:
            v_score_model = t_score_model = p_score_model = None 
        
        def find_score_token_index_by_prompt(prompt_text: str) -> int:
            prompt_tokens = self.tokenizer.encode(prompt_text, add_special_tokens=False)
            gen_ids = gen_token_ids  

            for i in range(len(gen_ids) - len(prompt_tokens)):
                if gen_ids[i:i+len(prompt_tokens)] == prompt_tokens:
                    j = i + len(prompt_tokens)
                    while j < len(gen_ids):
                        token_str = self.tokenizer.decode([gen_ids[j]], skip_special_tokens=True).strip()
                        if token_str.isdigit():
                            return j
                        j += 1
            return -1

        idx_v = find_score_token_index_by_prompt("(1) visual quality:")
        idx_t = find_score_token_index_by_prompt("(2) text-to-video alignment:")
        idx_p = find_score_token_index_by_prompt("(3) physical/common-sense consistency:")

        def compute_ll_based_soft_score(hard_val, token_idx):
            if hard_val is None or token_idx < 0:
                return None
            logits = scores[token_idx][0]  # [vocab]
            token_id = gen_token_ids[token_idx]
            logp = torch.log_softmax(logits, dim=-1)[token_id].item()
            prob = float(np.exp(logp))
            soft = hard_val * prob
            print(f"hard score={hard_val}, token_id={token_id}, logp={logp:.4f}, prob={prob:.4f}, soft score={soft:.4f}")
            return soft
        
        v_soft = compute_ll_based_soft_score(v_score_model, idx_v)
        t_soft = compute_ll_based_soft_score(t_score_model, idx_t)
        p_soft = compute_ll_based_soft_score(p_score_model, idx_p)

        return v_soft, t_soft, p_soft, output_text
        
    