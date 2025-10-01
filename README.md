# VideoScore2
This is the official repo for our paper: "VideoScore2: Think before You Score in Generative Video Evaluation"

<a target="_blank" href="https://arxiv.org/abs/2509.22799">
<img style="height:22pt" src="https://img.shields.io/badge/-Paper-red?style=flat&logo=arxiv"></a>
<a target="_blank" href="https://github.com/TIGER-AI-Lab/VideoScore2">
<img style="height:22pt" src="https://img.shields.io/badge/-Code-green?style=flat&logo=github"></a>
<a target="_blank" href="https://tiger-ai-lab.github.io/VideoScore2/">
<img style="height:22pt" src="https://img.shields.io/badge/-🌐%20Website-blue?style=flat"></a>
<a target="_blank" href="https://huggingface.co/datasets/TIGER-Lab/VideoFeedback2">
<img style="height:22pt" src="https://img.shields.io/badge/-🤗%20Dataset-red?style=flat"></a>
<a target="_blank" href="https://huggingface.co/TIGER-Lab/VideoScore2">
<img style="height:22pt" src="https://img.shields.io/badge/-🤗%20Models-red?style=flat"></a>
<a target="_blank" href="https://x.com/DongfuJiang/status/1973465380028031463">
<img style="height:22pt" src="https://img.shields.io/badge/-Post-black?style=flat&logo=x&logoColor=white"></a>
<br>

## Introduction
Recent advances in text-to-video generation have produced increasingly realistic and diverse content, yet evaluating such videos remains a fundamental challenge due to their multi-faceted nature encompassing visual quality, semantic alignment, and physical consistency. Existing evaluators and reward models are limited to single opaque scores, lack interpretability, or provide only coarse analysis, making them insufficient for capturing the comprehensive nature of video quality assessment. We present VideoScore2, a multi-dimensional, interpretable, and human-aligned framework that explicitly evaluates visual quality, text-to-video alignment, and physical/common-sense consistency while producing detailed chain-of-thought rationales. Our model is trained on a large-scale dataset VideoFeedback2 containing 27,168 human-annotated videos with both scores and reasoning traces across three dimensions, using a two-stage pipeline of supervised fine-tuning followed by reinforcement learning with Group Relative Policy Optimization (GRPO) to enhance analytical robustness. Extensive experiments demonstrate that VideoScore2 achieves superior performance with 44.35 (+5.94) accuracy on our in-domain benchmark VideoScore-Bench-v2 and 50.37 (+4.32) average performance across four out-of-domain benchmarks (VideoGenReward-Bench, VideoPhy2, etc), while providing interpretable assessments that bridge the gap between evaluation and controllable generation through effective reward modeling for Best-of-N sampling.

## Installation
🚧 TODO

## Dataset and Model
🚧 TODO

## Inference
🚧 TODO

## Training
🚧 TODO

## Evaluation
🚧 TODO

## Acknowledgement
This project builds upon several open-source frameworks:
- Thanks [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) for the SFT framework and codebase!
- Thanks [Video-R1](https://github.com/tulerfeng/Video-R1) for the Video RL framework and codebase!

## Citation
```bibtex
@misc{he2025videoscore2thinkscoregenerative,
      title={VideoScore2: Think before You Score in Generative Video Evaluation}, 
      author={Xuan He and Dongfu Jiang and Ping Nie and Minghao Liu and Zhengxuan Jiang and Mingyi Su and Wentao Ma and Junru Lin and Chun Ye and Yi Lu and Keming Wu and Benjamin Schneider and Quy Duc Do and Zhuofeng Li and Yiming Jia and Yuxuan Zhang and Guo Cheng and Haozhe Wang and Wangchunshu Zhou and Qunshu Lin and Yuanxing Zhang and Ge Zhang and Wenhao Huang and Wenhu Chen},
      year={2025},
      eprint={2509.22799},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2509.22799}, 
}

```
