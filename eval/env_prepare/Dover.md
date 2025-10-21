suppose you are in dir 'eval'
```
conda create -n dover python=3.10 -y --no-default-packages
conda activate dover
```
```
# shared packages for all baselines
pip install datasets==2.19.2 gdown opencv-python-headless pandas pyarrow 
```
```
git clone https://github.com/QualityAssessment/DOVER.git 
cd DOVER 
pip install -e .  
cd ..
rm -rf DOVER
```
```
mkdir -p eval_methods/utils_dover/pretrained_weights
cd eval_methods/utils_dover/pretrained_weights
wget https://github.com/QualityAssessment/DOVER/releases/download/v0.1.0/DOVER.pth 
wget https://github.com/QualityAssessment/DOVER/releases/download/v0.5.0/DOVER-Mobile.pth
cd ../../..
```