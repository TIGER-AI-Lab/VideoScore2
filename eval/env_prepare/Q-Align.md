suppose you are in dir 'eval'
```
conda create -n q_align python=3.10 -y --no-default-packages
conda activate q_align
```
```
# shared packages for all baselines
pip install datasets==2.19.2 gdown opencv-python-headless pandas pyarrow 
```
```
git clone https://github.com/Q-Future/Q-Align.git
cd Q-Align
pip install -e .
pip install numpy==1.26.4
pip install protobuf
cd ..
rm -rf Q-Align
```