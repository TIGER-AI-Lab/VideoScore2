from transformers import AutoProcessor, AutoModelForVision2Seq, AutoTokenizer
from qwen_vl_utils import process_vision_info
import torch
import numpy as np
import cv2, os, re
from typing import Tuple, List, Dict, Any

def _get_video_fps(url_or_p:str):
    cap = cv2.VideoCapture(url_or_p)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {url_or_p}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps

class eval_VideoScore2_float_v0:
    def __init__(self, model_name: str):
        self.model, self.processor = self.load_model_processor(model_name)

        # 尝试从 processor 拿 tokenizer
        self.tokenizer = getattr(self.processor, "tokenizer", None)
        if self.tokenizer is None:
            # 没有就单独加载
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                use_fast=False,
            )

        # ⚠️ 注意：candidate_token_ids 必须在 tokenizer 初始化后构建
        self.candidate_token_ids = self._build_numeric_candidates(num_min=0, num_max=10)

    def load_model_processor(self, model_name):
        model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            trust_remote_code=True,
        ).to("cuda")
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        return model, processor

    def _build_numeric_candidates(self, num_min=0, num_max=10):
        """
        构建 {整数值: token_id} 字典，只保留能用单 token 表示的数字
        """
        mapping = {}
        for n in range(num_min, num_max + 1):
            ids = self.tokenizer.encode(str(n), add_special_tokens=False)
            if len(ids) == 1:
                mapping[n] = ids[0]
        return mapping

    def _soft_score_from_step(self, step_logits: torch.Tensor) -> float:
        if not self.candidate_token_ids:
            return float("nan")
        probs = torch.softmax(step_logits, dim=-1)
        numer, denom = 0.0, 0.0
        for n, tid in self.candidate_token_ids.items():
            p = probs[tid].item()
            numer += n * p
            denom += p
        if denom == 0:
            return float("nan")
        return numer / denom

    def _fallback_scaled_score(self, hard_val: int, step_logits: torch.Tensor, token_id: int) -> float:
        logp = torch.log_softmax(step_logits, dim=-1)[token_id].item()
        return hard_val * float(np.exp(logp))
    
    def _group_digit_tokens(self, gen_token_ids):
        # 逐 token 解码（更稳）：得到人类可读字符
        decoded = [
            self.tokenizer.decode([tid], skip_special_tokens=True) for tid in gen_token_ids
        ]

        def is_digit_tok(s: str) -> bool:
            s = s.strip()
            return (s != "") and s.isdigit()

        results = []
        i = 0
        while i < len(decoded):
            if is_digit_tok(decoded[i]):
                start = i
                s = decoded[i].strip()
                j = i + 1
                while j < len(decoded) and is_digit_tok(decoded[j]):
                    s += decoded[j].strip()
                    j += 1
                try:
                    val = int(s)
                    results.append((start, j - 1, val))
                except ValueError:
                    pass
                i = j
            else:
                i += 1
        return results
    
    
    
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
                        "text": user_prompt,
                        # "text": "hello world, how are you?"
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
            temperature=0.6,
        )
        sequences = gen_out.sequences  
        scores = gen_out.scores        

        input_len = inputs["input_ids"].shape[1]
        def debug_print_token_score_mapping(input_len, sequences, scores, tokenizer):
            print("\n========= [Token ↔ LogProb Mapping] =========")
            generated = sequences[0][input_len:].tolist()
            for i, token_id in enumerate(generated):
                token_str = tokenizer.decode([token_id], skip_special_tokens=True)
                logits = scores[i][0]  # [vocab]
                if torch.allclose(logits, logits[0]):
                    print(f"[WARN] logits at step {i} are constant → invalid scores?")
                logp = torch.log_softmax(logits, dim=-1)[token_id].item()
                prob = float(np.exp(logp))
                print(f"[{input_len + i:03d}] (rel {i:02d}) token_id={token_id:5d} "
                    f"→ '{token_str:20}' | logp={logp:+.5f} | prob={prob:.5f}")
            print("=============================================\n")
        
        # debug_print_token_score_mapping(input_len, sequences, scores, self.tokenizer)
        
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

        digit_spans = self._group_digit_tokens(gen_token_ids)  
        
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
        print(idx_v)
        print(idx_t)
        print(idx_p)
        
        def compute_ll_based_soft_score_v0(hard_val, token_idx):
            if hard_val is None or token_idx < 0:
                return None
            relative_idx = token_idx - input_len
            if not (0 <= relative_idx < len(scores)):
                print(f"[warn] token_idx {token_idx} (rel {relative_idx}) out of range")
                return None
            logits = scores[relative_idx][0]  # [vocab]
            token_id = gen_token_ids[relative_idx]
            logp = torch.log_softmax(logits, dim=-1)[token_id].item()
            prob = float(np.exp(logp))
            soft = hard_val * prob
            print(f"[debug] score={hard_val}, token_id={token_id}, logp={logp:.4f}, prob={prob:.4f}, soft={soft:.4f}")
            return soft

        def compute_ll_based_soft_score(hard_val, token_idx):
            if hard_val is None or token_idx < 0:
                return None
            logits = scores[token_idx][0]  # [vocab]
            token_id = gen_token_ids[token_idx]
            logp = torch.log_softmax(logits, dim=-1)[token_id].item()
            prob = float(np.exp(logp))
            soft = hard_val * prob
            print(f"[debug] score={hard_val}, token_id={token_id}, logp={logp:.4f}, prob={prob:.4f}, soft={soft:.4f}")
            return soft
        
        v_soft = compute_ll_based_soft_score(v_score_model, idx_v)
        t_soft = compute_ll_based_soft_score(t_score_model, idx_t)
        p_soft = compute_ll_based_soft_score(p_score_model, idx_p)

        return v_soft, t_soft, p_soft, output_text
        
    