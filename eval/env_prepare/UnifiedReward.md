suppose you are in dir 'eval'
```
conda create -n unified_reward python=3.10 -y --no-default-packages
conda activate unified_reward
```
```
# shared packages for all baselines
pip install datasets==2.19.2 gdown opencv-python-headless pandas pyarrow 
```
```
git clone https://github.com/CodeGoat24/UnifiedReward.git
cd UnifiedReward
pip install --upgrade pip  
pip install -e ".[train]"
pip install setuptools
pip install flash_attn==2.5.8 --no-build-isolation
pip install certifi
cd ..
rm -rf UnifiedReward
```