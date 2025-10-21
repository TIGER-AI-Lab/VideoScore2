suppose you are in dir 'eval'
```
conda create -n videophy python=3.11 -y --no-default-packages
conda activate videophy
```
```
# shared packages for all baselines
pip install datasets==2.19.2 gdown opencv-python-headless pandas pyarrow 
```
```
git clone https://github.com/Hritikbansal/videophy.git
cd videophy
pip install -r requirements.txt
pip install transformers==4.46.1
cd ..
rm -rf videophy
```
```
# download ckpt to ./eval_methods/utils_video_phy2/checkpoints
mkdir -p eval_methods/utils_video_phy2/checkpoints
cd eval_methods/utils_video_phy2/checkpoints
conda install -c conda-forge git-lfs
git lfs install
git clone https://huggingface.co/videophysics/videophy_2_auto
cd ../../..
```