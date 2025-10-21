suppose you are in dir 'eval'
```
conda create -n deqa -y --no-default-packages
conda activate deqa
```
```
# shared packages for all baselines
pip install datasets==2.19.2 gdown opencv-python-headless pandas pyarrow 
```
```
git clone https://github.com/zhiyuanyou/DeQA-Score.git
cd DeQA-Score
pip install -e .
pip install requests
pip install transformers==4.36.1
pip install torch==2.0.1
pip install numpy==1.26.4
pip install protobuf
pip install opencv-python-headless==4.11.0.86
cd ..
rm -rf DeQA-Score
```