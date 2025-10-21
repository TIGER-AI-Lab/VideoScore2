suppose you are in dir 'eval'
```
conda create -n vision_reward python=3.11 -y --no-default-packages
conda activate vision_reward
```
```
# shared packages for all baselines
pip install datasets==2.19.2 gdown opencv-python-headless pandas pyarrow 
```
```
git clone https://github.com/zai-org/VisionReward.git
cd VisionReward
pip install -r requirements.txt
cd ..
rm -rf VisionReward
```