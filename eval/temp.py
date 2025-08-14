from transformers import AutoModelForVision2Seq, AutoProcessor
import torch
import numpy as np

# 设置模型名称
model_name = "Qwen/Qwen2.5-VL-7B-Instruct"

# 加载模型和 tokenizer（processor 中包含）
model = AutoModelForVision2Seq.from_pretrained(model_name, trust_remote_code=True).to("cuda")
processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
tokenizer = processor.tokenizer

# 输入问题（对纯文本使用 chat_template 即可）
prompt = "What is the capital of France?"

# 构造聊天格式输入
messages = [{"role": "user", "content": prompt}]
input_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# 编码输入
inputs = tokenizer([input_text], return_tensors="pt").to("cuda")
input_ids = inputs["input_ids"]
input_len = input_ids.shape[1]

# 生成答案，并开启 logits 输出
# with torch.no_grad():
#     output = model.generate(
#         **inputs,
#         max_new_tokens=50,
#         return_dict_in_generate=True,
#         output_scores=True,
#     )

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=50,
        return_dict_in_generate=True,
        output_scores=True,
        # 关键修改：启用采样
        do_sample=True,
        temperature=0.8, # 调整温度以控制随机性，更低的温度值意味着更少的随机性
    )

# 提取生成内容
sequences = output.sequences[0]  # [input_len + gen_len]
generated_ids = sequences[input_len:]  # only new tokens
scores = output.scores  # List of [1, vocab], length = gen_len

# 打印每个生成 token 的 logp
print("\n=== Generated Token Logprobs ===")
for i, token_id in enumerate(generated_ids):
    token_str = tokenizer.decode([token_id], skip_special_tokens=True).strip()
    logits = scores[i][0]  # [vocab]
    log_probs = torch.log_softmax(logits, dim=-1)
    logp = log_probs[token_id].item()
    prob = np.exp(logp)
    print(f"[{i:02d}] token_id={token_id:5d} → '{token_str:20}' | logp={logp:+.4f} | prob={prob:.4f}")
