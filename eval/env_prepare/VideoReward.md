suppose you are in dir 'eval'
```
# shared packages for all baselines
pip install datasets==2.19.2 gdown opencv-python-headless pandas pyarrow 
```
```
git clone https://github.com/KwaiVGI/VideoAlign
cd VideoAlign
conda env create -f environment.yaml
conda activate VideoReward
pip install flash-attn==2.5.8 --no-build-isolation
cd ..
rm -rf VideoAlign
```
```
mkdir -p eval_methods/utils_video_reward/checkpoints
cd eval_methods/utils_video_reward/checkpoints
git lfs install
git clone https://huggingface.co/KwaiVGI/VideoReward
pip install numpy
```