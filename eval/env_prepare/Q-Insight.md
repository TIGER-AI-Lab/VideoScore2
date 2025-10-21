suppose you are in dir 'eval'
```
conda create -n q_insight python=3.10 -y --no-default-packages
conda activate q_insight
```
```
# shared packages for all baselines
pip install datasets==2.19.2 gdown opencv-python-headless pandas pyarrow 
```
```
git clone https://github.com/bytedance/Q-Insight.git
cd Q-Insight
cd src/open-r1-multimodal 
pip install -e ".[dev]"
pip install wandb==0.18.3
pip install tensorboardx
pip install qwen_vl_utils torchvision
pip install flash-attn --no-build-isolation
pip install transformers==4.51.3
cd ../../..
rm -rf Q-Insight
```