import io
import json
import numpy as np
import torch
from decord import cpu, VideoReader, bridge
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from typing import Optional

class eval_VisionReward:
    def __init__(
        self,
        model_path: str = "THUDM/VisionReward-Video",
        questions_path: str = "eval_methods/utils_vision_reward/VisionReward_video_qa_select.txt",
        weight_path: str = "eval_methods/utils_vision_reward/weight.json"
    ):
        self.model_path = model_path
        self.questions_path = questions_path
        self.weight_path = weight_path

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.torch_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16

        # Load resources
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=self.torch_dtype,
            trust_remote_code=True
        ).eval().to(self.device)

        with open(self.questions_path, 'r') as f:
            self.questions = f.readlines()
            
        with open(self.weight_path, 'r') as f:
            weight = json.load(f)
            self.weight = np.array(weight)

    def _load_video(self, video_path: str, strategy: str = 'chat', num_frames: int = 24):
        bridge.set_bridge('torch')
        with open(video_path, 'rb') as f:
            video_bytes = f.read()

        decord_vr = VideoReader(io.BytesIO(video_bytes), ctx=cpu(0))
        total_frames = len(decord_vr)

        if strategy == 'base':
            start_frame = 0
            end_frame = min(total_frames, int(60 * decord_vr.get_avg_fps()))
            frame_ids = np.linspace(start_frame, end_frame - 1, num_frames, dtype=int)
        else:  # chat strategy
            timestamps = decord_vr.get_frame_timestamp(np.arange(total_frames))
            timestamps = [i[0] for i in timestamps]
            max_second = round(max(timestamps)) + 1
            frame_ids = []
            for second in range(max_second):
                closest = min(timestamps, key=lambda x: abs(x - second))
                index = timestamps.index(closest)
                frame_ids.append(index)
                if len(frame_ids) >= num_frames:
                    break

        video_tensor = decord_vr.get_batch(frame_ids)
        video_tensor = video_tensor.permute(3, 0, 1, 2)
        return video_tensor

    def _inference(self, video_tensor: torch.Tensor, query: str, temperature: float = 0.1) -> str:
        history = []
        inputs = self.model.build_conversation_input_ids(
            tokenizer=self.tokenizer,
            query=query,
            images=[video_tensor],
            history=history,
            template_version='chat'
        )
        inputs = {
            'input_ids': inputs['input_ids'].unsqueeze(0).to(self.device),
            'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to(self.device),
            'attention_mask': inputs['attention_mask'].unsqueeze(0).to(self.device),
            'images': [[inputs['images'][0].to(self.device).to(self.torch_dtype)]],
        }
        gen_kwargs = {
            "max_new_tokens": 2048,
            "pad_token_id": 128002,
            "top_k": 1,
            "do_sample": False,
            "top_p": 0.1,
            "temperature": temperature,
        }
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
            outputs = outputs[:, inputs['input_ids'].shape[1]]
        return self.tokenizer.decode(outputs[0]).strip().lower()

    def evaluate_video(self, prompt: str, video_path: str, kwargs: dict) -> float:
        """
        Evaluate video quality for the given prompt using VisionReward model.
        Returns: weighted score ∈ [-1, 1]
        """
        video_tensor = self._load_video(video_path)
        queries = [q.replace('[[prompt]]', prompt) for q in self.questions]
        
        answers = []
        for query in tqdm(queries, desc='Evaluating VisionReward'):
            answer = self._inference(video_tensor, query)
            
            answers.append(answer)
        binary_answers = np.array([1 if "yes" in ans else -1 for ans in answers])
        score = np.mean(binary_answers * self.weight).item()
        return score,score,score,None
