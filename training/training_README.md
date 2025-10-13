# Stage-1: SFT (Supervised Fine-Tuning)
We perform Supervised Fine-Tuning (SFT) within the [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) framework.
Follow the steps below to set up your environment and run training with the VideoScore2 SFT dataset.

### 1️⃣ Clone from the original or your forked version
```
git clone <url of llama-factory or your forked one>
```

### 2️⃣ Create a Dedicated Environment
It’s recommended to create a clean environment (using conda or uv) before installing dependencies.
```
# using conda
conda create -n lmfac python=3.10 -y
conda activate lmfac

# OR using uv
uv venv --python=3.10 .envs_lmfac
source .envs_lmfac/bin/activate
```

### 3️⃣ Install Dependencies
```
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
pip install wandb
pip install deepspeed==0.16.9
pip install --no-deps transformers==4.50.0
```

### 4️⃣ Prepare Data and Config Files
- Copy 'SFT/prepare_SFT_data.py' in our repo to 'LLaMA-Factory/'

- Copy 'SFT/vs2_qwen2_5vl_sft_27k_5e-5_2fps_960_720_8192' in our repo to 'LLaMA-Factory/examples/train_full/'

- Prepare json data and videos: 
```bash
python prepare_sft_data.py --data_version_name data_27k_train_SFT
```

### 5️⃣ Final Directory Structure
```
LLaMA-Factory/
├── data/
│   ├── videos/  ## folder for videos of our dataset
│   ├── ...
│   ├── data_27k_train_SFT.json  #  SFT data of our dataset
│   ├── dataset_info.json  #  meta-info for different datasets
│   └── ...     
|                   
├── examples/                     
│   ├── train_full/                
│   │   ├── vs2_qwen2_5vl_sft_27k_5e-5_2fps_960_720_8192.yaml  # our SFT configs
│   │   ├── xxx.yaml
│   │   └── ...
│   ├── train_lora/                
│   └── ...                        
│
├── saves/  # checkpoint save dir                 
├── ...
├── prepare_SFT_data.py
├── requirements.txt               
├── README.md  
...
```

### 6️⃣ Launch Training
Set up environment variables and start SFT training (see SFT/run_sft.sh for reference):
```bash
export HF_HOME=<your_hf_cache_dir>
export HF_TOKEN=<your_hf_token>
export WANDB_API_KEY=<your_wandb_key>

wandb login --relogin $WANDB_API_KEY
llamafactory-cli train examples/train_full/vs2_qwen2_5vl_sft_27k_5e-5_2fps_960_720_8192.yaml \
    hf_hub_token=$HF_TOKEN \
    dataset=data_27k_train_sft
```
The checkpoint will be saves in 'LLaMA-Factory/saves/<run_name>'


# Stage-2: RL (Reinforcement Learning)
### 1️⃣ Clone from the original or your forked version
```
git clone <url of llama-factory or your forked one>
```

### 2️⃣ Create a Dedicated Environment
It’s recommended to create a clean environment (using conda or uv) before installing dependencies.
```
# using conda
conda create -n video_r1 python=3.11 --no-default-packages -y
conda activate video_r1

# OR using uv
uv venv --python=3.11 .envs_video_r1
source .envs_video_r1/bin/activate
```

### 3️⃣ Install Dependencies
```
cd Video-R1
bash setup.sh
cd src/qwen-vl-utils
pip install -e .[decord]
cd ../..
```
As mentioned in the original repo, since Qwen2.5-VL has been frequently updated in the Transformers library, which may cause version-related bugs or inconsistencies. The code of Video-R1 is compatible with the following version, please download at [google-drive](https://drive.google.com/file/d/1Kc81WZitEhUZYWXpL6y2GXuSXufLSYcF/view?usp=sharing)

```
unzip transformers-main.zip
cd ./transformers-main
pip install .
cd ..
```
For vLLM library, please use 0.7.2 version; For trl library, please use 0.16.0 version.

(All the information above can be found in the Video-R1 repository.)

Ensure that PyTorch and FlashAttention-2 are properly installed and working by running: 
```bash
python -c "import torch"
python -c "import flash_attn_2_cuda"
```

### 4️⃣ Prepare Data and Config Files
- Copy 'RL/prepare_RL_data.py' in our repo to 'Video-R1/src'

- Copy 'RL/grpo_vs2_sft.py' in our repo to 'Video-R1/src/r1-v/src/open_r1/'

- Copy 'RL/grpo_vs2_no_sft.py' in our repo to 'Video-R1/src/r1-v/src/open_r1/'

- Copy 'RL/run_grpo_with_sft.sh' in our repo to 'Video-R1/scripts/'

- Copy 'RL/run_grpo_wo_sft.sh' in our repo to 'Video-R1/scripts/'

- Prepare json data and videos: 
```bash
python prepare_rl_data.py \
  --json_name data_27k_train_RL \
  --video_zip_name videos_27k \
  --data_save_dir "r1-v/Video-R1-data" 
```

### 5️⃣ Final Directory Structure
```
Video-R1/
├── src/
│   ├── r1-v/
│   │   ├── configs/
│   │   ├── log/   
│   │   │   ├── <run_name>/
│   │   │   └── ...
│   │   ├── src/
│   │   │   └── open_r1/
│   │   │       ├── trainer/
│   │   │       │   ├── ...
│   │   │       │   ├── __init__.py
│   │   │       │   ├── grpo_trainer.py
│   │   │       │   └── grpo_trainer_vs2.py
│   │   │       │
│   │   │       ├── __init__.py
│   │   │       ├── grpo.py
│   │   │       ├── grpo_vs2_sft.py
│   │   │       ├── grpo_vs2_no_sft.py
│   │   │       └── ...
│   │   ├── Video-R1-data/
│   │   │   ├── vs2_videos/
│   │   │   └── data_27k_rl_train.json
│   │   ├── wandb/
│   │   ├── prepare_rl_data.py
│   │   └── ...
│   │
│   ├── scripts/
│   │   ├── run_grpo_with_sft.sh
│   │   ├── run_grpo_wo_sft.sh
│   │   └── ...
│   │
│   └── qwen-vl-utils
│
├── setup.sh
├── ...


```

### 6️⃣ Launch Training
Set up environment variables and start RL training:
```bash
export HF_HOME=<your_hf_cache_dir>
export HF_TOKEN=<your_hf_token>
export WANDB_API_KEY=<your_wandb_key>

bash src/scripts/run_grpo_with_sft.sh
# or for ablatiion of RL w/o SFT
# bash src/scripts/run_grpo_wo_sft.sh
```
The checkpoint will be saves in 'Video-R1/src/r1-v/logs/<run_id>'