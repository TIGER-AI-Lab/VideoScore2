suppose you are in dir 'eval'
```
conda create -n image_reward python=3.10 -y --no-default-packages
conda activate image_reward
```
```
# shared packages for all baselines
pip install datasets==2.19.2 gdown opencv-python-headless pandas pyarrow 
```
```
pip install image-reward
pip install clip
```