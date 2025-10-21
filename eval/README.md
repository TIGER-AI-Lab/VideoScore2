## Prepare Baselines (build env and ckpt)
We have two classes of baselines: 
- (1) MLLM Prompting Methods (via [OpenRouter](https://openrouter.ai/) API calling)
- (2) reward/scoring models for vision, both image and video

Different baseline reward models requires different dependencies, please create a separate environment, then install dependencies and download checkpoint (if needed) as shown in text files in dir `eval/env_prepare/`. 

Here we take baseline 'VideoPhy2-auto-eval' as an example: 

(1) clone original repo (suppose you are in dir `eval` now)
```
# suppose you are in dir 'eval'
git clone https://github.com/Hritikbansal/videophy.git
```
(2) create separate environment with conda or uv
```
# using conda
cd videophy
conda create -n videophy python=3.10 -y
conda activate videophy

# OR using uv
mkdir -p .envs
uv venv --python=3.10 .envs/videophy
source .envs/videophy/bin/activate
```
(3) install dependencies
```
pip install -r requirements.txt
pip install transformers==4.46.1
pip install datasets==2.19.2 gdown opencv-python-headless pandas pyarrow
```
(4) download model checkpoint of 'VideoPhy2-auto-eval'
```
# suppose you are in dir 'eval/videophy' now
cd ..
mkdir -p eval_methods/utils_video_phy2
cd eval_methods/utils_video_phy2
mkdir -p checkpoints
cd checkpoints
conda install -c conda-forge git-lfs
git lfs install
git clone https://huggingface.co/videophysics/videophy_2_auto
cd ../../..
```

For other baselines, refer to corresponding `.txt` file in dir `eval/env_prepare/`.

## Run Baselines (including our model VideoScore2)
🚧TODO

## Prepare Benchmark Data
When you run the evaluation script, the corresponding benchmark data will be downloaded automatically.

If you prefer to download and process the benchmark data separately, you can do so with the following command:

```
python benchmark.py --bench_name "vs2_bench" --loaded_num "all"
```
Note: 
 - for `bench_name`, we support the in-domain benchmark (ours) `VideoScore-Bench-v2` and 4 out-of-domain benchmark: `VideoGen-Reward-Bench`, `T2VQA-DB`, `MJ-Bench-Video` and `VideoPhy2-test`, as reported in the paper.
 - for `loaded_num`, choose 'all' to load the full dataset, or specify any integer not larger than the dataset size.
 ```
 # size of each benchmark
 {
    "vs2_bench":500,
    "videogen_reward_bench":4691,
    "t2vqa_db":2000,
    "mj_bench_video":2170,
    "video_phy2_test"3396,
 }
 ```

## Metric Calculation
🚧TODO
