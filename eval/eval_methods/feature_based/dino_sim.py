import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2

def compute_dino_similarity(model, video_path: str, prompt: str = "") -> float:
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # 3. Extract representative frames
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count < 1:
        return -1.0

    frame_idxs = np.linspace(0, frame_count - 1, num=4, dtype=int)
    feats = []

    for idx in frame_idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        success, frame = cap.read()
        if not success:
            continue
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img_tensor = preprocess(img).unsqueeze(0).to("cuda")

        with torch.no_grad():
            feat = model(img_tensor).squeeze().cpu().numpy()
        feats.append(feat)

    cap.release()

    if len(feats) < 2:
        return -1.0
    feats = np.stack(feats)
    sim = np.mean([
        np.dot(feats[i], feats[j]) / (np.linalg.norm(feats[i]) * np.linalg.norm(feats[j]) + 1e-8)
        for i in range(len(feats)) for j in range(i + 1, len(feats))
    ])
    return float(sim)
