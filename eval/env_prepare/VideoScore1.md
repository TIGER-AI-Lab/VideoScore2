suppose you are in dir 'eval'
```
conda create -n vs1_eval python=3.11 -y --no-default-packages
conda activate vs1_eval
```
```
# shared packages for all baselines
pip install datasets==2.19.2 gdown opencv-python-headless pandas pyarrow 
```
```
pip install transformers==4.45.0
pip install qwen-vl-utils
pip install accelerate
pip install scipy
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0
pip install mantis-vl
```