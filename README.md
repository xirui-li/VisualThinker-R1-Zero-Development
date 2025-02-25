<div align="center">

# [Place Holder]

[![Notion](https://img.shields.io/badge/Notion-%23000000.svg?style=for-the-badge&logo=notion&logoColor=white)](https://hkust-nlp.notion.site/simplerl-reason) [![Hugging Face](https://img.shields.io/badge/SimpleRL-fcd022?style=for-the-badge&logo=Huggingface&logoColor=000)](https://huggingface.co/collections/hkust-nlp/simplerl-67b543892b2ec6908ffff710)

</div>


This repo contains a simple reinforcement learning recipe to improve models' reasoning abilities. It is simple because only rule-based reward is used, the recipe is almost the same as the one used in [DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1), except that the code currently uses PPO rather than GRPO. We have used this code to train small models (7B) on limited data (8K examples), achieving surprisingly strong results -- for example, starting from Qwen2.5-Math-7B (base model), we perform RL on it directly. No SFT, no reward model, just 8K MATH examples for verification, the resultant model achieves (pass@1) 33.3% on AIME, 62.5% on AMC, and 77.2% on MATH, outperforming Qwen2.5-math-7B-instruct and being comparable to previous baselines that use >50x more data and more complicated components. You may check our Notion blog or the Introduction below for more details.  

<div align="center">
<img src="https://github.com/user-attachments/assets/bacd1680-ccb0-4921-a687-8a595ebf5896" width="700" alt="simplelr-reaoning-intro-figure_00">
</div>

> Training dynamics of our Qwen2.5-SimpleRL-Zero training starting from the Qwen2.5-Math-7B, without SFT or reward models.

![Reinforcement Learning](https://img.shields.io/badge/Algo-Reinforcement--Learning-red) 
![R1](https://img.shields.io/badge/Algo-R1-red) 
![Vision-Centric](https://img.shields.io/badge/Task-Vision--Perception-yellow) 
![Qwen-VL](https://img.shields.io/badge/Model-Qwen--VL-green)
![Aha-Moment](https://img.shields.io/badge/Analysis-Aha--moment-green) 

[PLACE HOLDER] is a reproduction of DeepSeek R1 Zero in vision centric tasks. We built upon Open-R1-Multimodal and R1-V.

Through applying GRPO on the 2B base LM develops self-verification autonomously and exhibits an emergent ability to "take another look" at the image and correct its mistakes.

**TL;DR:**
1. We are the **first to replicate a key characteristic** of R1 success (**”aha moment”** and **increasing reasoning length**) on **multimodal** reasoning tasks.

2. We showed that **vision-centric** tasks could also benefit from improved reasoning capabilities.

   
Paper's on it's way, stay tuned!

**Twitter thread: ** [PLACE HOLDER]

**Full experiment log:** [PLACE HOLDER]

**Blogs:** [PLACE HOLDER]

### Updates:
- 2025-02-24: We release the [PLACE HOLDER] repo.


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
sh run_grpo_SAT.sh # Adjust open-r1-multimodal/configs/zero3.yaml or zero2.yaml accordingly
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

## Acknowledgements

We sincerely thank [DeepSeek](https://github.com/deepseek-ai/DeepSeek-R1), [Open-R1](https://github.com/huggingface/open-r1), [QwenVL](https://github.com/QwenLM/Qwen2.5-VL), [Open-R1-Multimodal](https://github.com/EvolvingLMMs-Lab/open-r1-multimodal), [R1-V](https://github.com/Deep-Agent/R1-V), [SAT](https://arxiv.org/abs/2412.07755), [CV-Bench](https://cambrian-mllm.github.io/) for providing open source resources and to build the project. 

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

