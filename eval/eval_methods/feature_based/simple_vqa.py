import torch
import torch.nn as nn
from eval_methods.feature_based.utils_simple_vqa import UGC_BVQA_model
from pytorchvideo.models.hub import slowfast_r50
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
from typing import Optional
from eval_methods.utils import extract_video_frame_imgs  # optional if reused
import os

model_pth="eval_methods/feature_based/utils_simple_vqa/UGC_BVQA_model.pth"

class SlowFastFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = slowfast_r50(pretrained=True)
        features = list(backbone.children())[0]
        self.feature_extraction = nn.Sequential(*features[:5])
        self.slow_avg_pool = features[5].pool[0]
        self.fast_avg_pool = features[5].pool[1]
        self.adp_avg_pool = features[6].output_pool

    def forward(self, x):
        with torch.no_grad():
            x = self.feature_extraction(x)
            slow_feat = self.slow_avg_pool(x[0])
            fast_feat = self.fast_avg_pool(x[1])
            slow_feat = self.adp_avg_pool(slow_feat).squeeze()
            fast_feat = self.adp_avg_pool(fast_feat).squeeze()
        return torch.cat([slow_feat, fast_feat], dim=-1)


def pack_pathway_output(frames, device):
    slow_pathway = torch.index_select(
        frames,
        2,
        torch.linspace(0, frames.shape[2] - 1, frames.shape[2] // 4).long(),
    )
    return [slow_pathway.to(device), frames.to(device)]


def simple_vqa_output(model_or_process, video_path: str, prompt: str = "") -> Optional[float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----- Load model -----
    model_motion = SlowFastFeatureExtractor().to(device)
    model_quality = UGC_BVQA_model.resnet50(pretrained=False)
    model_quality = torch.nn.DataParallel(model_quality).to(device)
    model_quality.load_state_dict(torch.load(model_pth))
    model_quality.eval()

    # ----- Process spatial stream -----
    cap = cv2.VideoCapture(video_path)
    frame_rate = int(round(cap.get(cv2.CAP_PROP_FPS)))
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    num_clips = int(num_frames / frame_rate)
    num_clips = max(num_clips, 8)  # at least 8

    spatial_transform = transforms.Compose([
        transforms.Resize(520),
        transforms.CenterCrop(448),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    spatial_frames = []

    read_idx = 0
    for i in range(num_frames):
        has_frame, frame = cap.read()
        if not has_frame:
            break
        if i % frame_rate == 0:
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            tensor = spatial_transform(img)
            spatial_frames.append(tensor)
            read_idx += 1

    # Padding if insufficient frames
    while len(spatial_frames) < num_clips:
        spatial_frames.append(spatial_frames[-1].clone())

    spatial_video = torch.stack(spatial_frames[:num_clips]).to(device).unsqueeze(0)

    # ----- Process motion stream -----
    motion_transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize([0.45, 0.45, 0.45], [0.225, 0.225, 0.225])
    ])
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    motion_frames = []

    for _ in range(num_frames):
        has_frame, frame = cap.read()
        if not has_frame:
            break
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        tensor = motion_transform(img)
        motion_frames.append(tensor)

    cap.release()

    if len(motion_frames) < 32:
        motion_frames += [motion_frames[-1]] * (32 - len(motion_frames))

    motion_clips = []
    for i in range(num_clips):
        start = i * frame_rate
        end = start + 32
        clip = motion_frames[start:end]
        if len(clip) < 32:
            clip += [clip[-1]] * (32 - len(clip))
        motion_clips.append(torch.stack(clip))

    while len(motion_clips) < 8:
        motion_clips.append(motion_clips[-1].clone())

    # ----- Feature extraction -----
    features_motion = []
    for clip in motion_clips:
        clip = clip.unsqueeze(0).permute(0, 2, 1, 3, 4)  # [B,C,T,H,W]
        motion_feat = model_motion(pack_pathway_output(clip[0], device))
        features_motion.append(motion_feat)

    motion_feature = torch.stack(features_motion).unsqueeze(0)  # [1, N, D]

    # ----- Predict score -----
    with torch.no_grad():
        score = model_quality(spatial_video, motion_feature).item()
        return float(score)
