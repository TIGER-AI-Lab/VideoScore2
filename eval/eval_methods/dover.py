import torch
import os
import pickle as pkl
import numpy as np
import yaml
import decord
from typing import Tuple
from eval_methods.utils_dover.datasets import UnifiedFrameSampler, spatial_temporal_view_decomposition
from eval_methods.utils_dover.models import DOVER

mean = torch.FloatTensor([123.675, 116.28, 103.53])
std = torch.FloatTensor([58.395, 57.12, 57.375])

class eval_DOVER:
    def __init__(self, config_path: str = "./eval_methods/utils_dover/dover.yml", device: str = 'cuda'):
        self.device = device
        self.config_path = config_path

        with open(config_path, "r") as f:
            self.opt = yaml.safe_load(f)

        # Load DOVER model
        test_load_path="./eval_methods/utils_dover/pretrained_weights/DOVER.pth"
        self.model = DOVER(**self.opt["model"]["args"]).to(self.device)
        self.model.load_state_dict(
            torch.load(test_load_path, map_location=self.device)
        )
        self.model.eval()

        # Prepare temporal samplers
        self.temporal_samplers = {}
        dopt = self.opt["data"]["val-l1080p"]["args"]
        for stype, sopt in dopt["sample_types"].items():
            if "t_frag" not in sopt:
                self.temporal_samplers[stype] = UnifiedFrameSampler(
                    sopt["clip_len"], sopt["num_clips"], sopt["frame_interval"]
                )
            else:
                self.temporal_samplers[stype] = UnifiedFrameSampler(
                    sopt["clip_len"] // sopt["t_frag"],
                    sopt["t_frag"],
                    sopt["frame_interval"],
                    sopt["num_clips"],
                )
        self.sample_types = dopt["sample_types"]

    def fuse_score(self, tqe_score: float, aqe_score: float) -> float:
        # Fusion using predefined weights and sigmoid normalization
        x = (tqe_score - 0.1107) / 0.07355 * 0.6104 + (aqe_score + 0.08285) / 0.03774 * 0.3896
        return 1 / (1 + np.exp(-x))

    def evaluate_video(self, prompt: str, video_path: str, kwargs: dict = None) -> Tuple[float, float, float, str]:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        # Decompose spatial-temporal views
        views, _ = spatial_temporal_view_decomposition(
            video_path, self.sample_types, self.temporal_samplers
        )

        for k, v in views.items():
            num_clips = self.sample_types[k].get("num_clips", 1)
            views[k] = (
                ((v.permute(1, 2, 3, 0) - mean) / std)
                .permute(3, 0, 1, 2)
                .reshape(v.shape[0], num_clips, -1, *v.shape[2:])
                .transpose(0, 1)
                .to(self.device)
            )

        with torch.no_grad():
            raw_scores = self.model(views)
            tqe_score = raw_scores[0].mean().item()
            aqe_score = raw_scores[1].mean().item()
        score = self.fuse_score(tqe_score, aqe_score)
        return score, score, score, f"Technical Quality: {tqe_score:.4f}, Aesthetic Quality: {aqe_score:.4f}"

