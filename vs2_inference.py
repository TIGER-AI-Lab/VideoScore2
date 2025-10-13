from transformers import AutoProcessor, AutoModelForVision2Seq, AutoTokenizer
from qwen_vl_utils import process_vision_info
import torch
import numpy as np
import cv2 
import os
import re
from string import Template

def _get_video_fps(url_or_p: str):
    cap = cv2.VideoCapture(url_or_p)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {url_or_p}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps

def ll_based_soft_score_normed(hard_val, token_idx, scores, tokenizer):
    if hard_val is None or token_idx < 0:
        return None
    logits = scores[token_idx][0]
    score_range = list(range(1, 6))
    score_probs = []
    for s in score_range:
        ids = tokenizer.encode(str(s), add_special_tokens=False)
        if len(ids) == 1:
            tid = ids[0]
            logp = torch.log_softmax(logits, dim=-1)[tid].item()
            prob = float(np.exp(logp))
            score_probs.append((s, prob))
    if not score_probs:
        print("[warn] No valid score token found.")
        return None
    scores_list, probs_list = zip(*score_probs)
    total_prob = sum(probs_list)
    max_prob = max(probs_list)
    best_score = scores_list[probs_list.index(max_prob)]
    normalized_prob = max_prob / total_prob if total_prob > 0 else 0
    return round(best_score * normalized_prob, 4)

def find_score_token_index_by_prompt(prompt_text, tokenizer, gen_ids):
    gen_str = tokenizer.decode(gen_ids, skip_special_tokens=False)
    pattern = r"(?:\(\d+\)\s*|\n\s*)?" + re.escape(prompt_text)
    match = re.search(pattern, gen_str, flags=re.IGNORECASE)
    if not match:
        return -1
    after_text = gen_str[match.end():]
    num_match = re.search(r"\d", after_text)
    if not num_match:
        return -1
    target_substr = gen_str[:match.end() + num_match.start() + 1]
    for i in range(len(gen_ids)):
        partial = tokenizer.decode(gen_ids[:i+1], skip_special_tokens=False)
        if partial == target_substr:
            return i
    return -1

def main(MODEL_NAME, video_path, t2v_prompt):
    # --- prepare query ---
    VS2_QUERY_TEMPLATE = Template("""
    You are an expert for evaluating AI-generated videos from three dimensions:
    (1) visual quality – clarity, smoothness, artifacts;
    (2) text-to-video alignment – fidelity to the prompt;
    (3) physical/common-sense consistency – naturalness and physics plausibility.

    Video prompt: $t2v_prompt

    Please output in this format:
    visual quality: <v_score>; 
    text-to-video alignment: <t_score>, 
    physical/common-sense consistency: <p_score>
    """)
    user_prompt = VS2_QUERY_TEMPLATE.substitute(t2v_prompt=t2v_prompt)

    if not os.path.exists(video_path):
        raise ValueError(f"Video not found: {video_path}")

    infer_fps, max_tokens, temperature = 2.0, 1024, 0.7
    if infer_fps == "raw":
        infer_fps = _get_video_fps(video_path)

    print(f"[Init] Loading model: {MODEL_NAME}")
    model = AutoModelForVision2Seq.from_pretrained(MODEL_NAME, trust_remote_code=True).to("cuda")
    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer = getattr(processor, "tokenizer", None) or AutoTokenizer.from_pretrained(
        MODEL_NAME, trust_remote_code=True, use_fast=False
    )

    # --- preprocess ---
    messages = [{"role": "user", "content": [
        {"type": "video", "video": video_path, "fps": infer_fps},
        {"type": "text", "text": user_prompt}
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        fps=infer_fps, padding=True, return_tensors="pt"
    ).to("cuda")

    # --- inference ---
    gen_out = model.generate(
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

    output_text = processor.batch_decode(
        sequences[:, input_len:], skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    print("\n[Raw Model Output]\n", output_text)

    # --- parse scores ---
    pattern = r"visual quality:\s*(\d+).*?text-to-video alignment:\s*(\d+).*?physical/common-sense consistency:\s*(\d+)"
    match = re.search(pattern, output_text, re.DOTALL | re.IGNORECASE)
    v_score = int(match.group(1)) if match else None
    t_score = int(match.group(2)) if match else None
    p_score = int(match.group(3)) if match else None

    idx_v = find_score_token_index_by_prompt("visual quality:", tokenizer, gen_token_ids)
    idx_t = find_score_token_index_by_prompt("text-to-video alignment:", tokenizer, gen_token_ids)
    idx_p = find_score_token_index_by_prompt("physical/common-sense consistency:", tokenizer, gen_token_ids)

    v_soft = ll_based_soft_score_normed(v_score, idx_v, scores, tokenizer)
    t_soft = ll_based_soft_score_normed(t_score, idx_t, scores, tokenizer)
    p_soft = ll_based_soft_score_normed(p_score, idx_p, scores, tokenizer)

    print("\n====== Inference Result ======")
    print(f"Video Path: {video_path}")
    print(f"Visual Quality: {v_soft}")
    print(f"Text-to-Video Alignment: {t_soft}")
    print(f"Physical Consistency: {p_soft}")
    print("==============================\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="VideoScore2 Inference Script")
    parser.add_argument("--video_path", type=str, required=True, help="Path to input video file")
    parser.add_argument("--t2v_prompt", type=str, required=True, help="Text prompt used to generate the video")
    parser.add_argument("--model_name", type=str, default="TIGER-Lab/VideoScore2", help="Model name or path")
    args = parser.parse_args()

    main(args.model_name, args.video_path, args.t2v_prompt)