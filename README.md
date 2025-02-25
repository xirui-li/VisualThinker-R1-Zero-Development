# [PLACE HOLDER]

![Reinforcement Learning](https://img.shields.io/badge/Algo-Reinforcement--Learning-red) 
![R1](https://img.shields.io/badge/Algo-R1-red) 
![Vision-Centric](https://img.shields.io/badge/Task-Vision--Perception-yellow) 
![Qwen-VL](https://img.shields.io/badge/Model-Qwen--VL-green)
![Aha-Moment](https://img.shields.io/badge/Analysis-Aha--moment-green) 

[PLACE HOLDER] is a reproduction of DeepSeek R1 Zero in vision centric tasks. We built upon veRL and R1-V.

Through simple RL Receip, the 2B base LM develops self-verification all on its own and exhibits an emergent ability to "take another look" at the image and correct its mistakes.

Twitter thread: [PLACE HOLDER]

Full experiment log: [PLACE HOLDER]

Paper's on it's way, stay tuned!

## Setup

```bash
bash setup.sh
```
## Prepare Dataset

```bash
cd src/data/SAT
bash prepare_dataset.sh
```

## Training

### GRPO Training
To reproduce checkpoing on different dataset, replace the sh file in the following command with run_grpo_{dataset_name}.sh
```bash
cd src/open-r1-multimodal
sh run_grpo_SAT.sh # Adjust open-r1-multimodal/configs/zero3.yaml or zero2.yaml accordingly # Full training for 2 epochs take more than 50 hours, we usually can observe reward peak and stop at step 100~500
```

### SFT Training

```bash
cd src/open-r1-multimodal
sh run_sft.sh # Adjust open-r1-multimodal/configs/zero3.yaml or zero2.yaml accordingly
```

## Evaluation

### CVBench Evaluation
```bash
cd src/eval
python test_qwen2vl_CVBench.py 
```

> [!NOTE] 
> 1. To reproduce the result, keep the per_device_train_batch_size to 1 for now, as there is a revealed bug about batched training. See the [reproduction report](https://github.com/Deep-Agent/R1-V/issues/4#issuecomment-2633348354) here. We realize it is important for effiency and are working on solving it with the community.
> 2. If you meet **OOM Error**, add `--deepspeed local_scripts/zero3.json` following https://github.com/Deep-Agent/R1-V/issues/18 or you can reduce `--num_generations`


## :coffee: Stay Connected!

We are always open to engaging discussions, collaborations, or even just sharing a virtual coffee. To get in touch or join our team, visit [TurningPoint AI](https://www.turningpoint-ai.com/)'s homepage for contact information.


## :white_check_mark: Cite

If you find our research useful for your your research and applications, please kindly cite using this BibTeX: *placeholder*

```latex
@misc{li2024mossbenchmultimodallanguagemodel,
      title={MOSSBench: Is Your Multimodal Language Model Oversensitive to Safe Queries?}, 
      author={Xirui Li and Hengguang Zhou and Ruochen Wang and Tianyi Zhou and Minhao Cheng and Cho-Jui Hsieh},
      year={2024},
      eprint={2406.17806},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2406.17806}, 
}
```

