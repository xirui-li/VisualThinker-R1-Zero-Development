<div align="center">

# [Place Holder]

[![Notion](https://img.shields.io/badge/Notion-%23000000.svg?style=for-the-badge&logo=notion&logoColor=white)](https://hkust-nlp.notion.site/simplerl-reason)

</div>


[PLACE HOLDER] is a reproduction of DeepSeek R1 Zero in vision centric tasks. We built upon Open-R1-Multimodal and R1-V.

Through applying GRPO on the 2B base LM develops self-verification autonomously and exhibits an emergent ability to "take another look" at the image and correct its mistakes.

1. We are the **first to replicate a key characteristic** of R1 success (**”aha moment”** and **increasing reasoning length**) on **multimodal** reasoning tasks.

2. We showed that **vision-centric** tasks could also benefit from improved reasoning capabilities.  

<div align="center">
<img src="https://multimodal-r1.s3.us-west-1.amazonaws.com/Training_Steps.png" width="700" alt="simplelr-reaoning-intro-figure_00">
</div>

> Training dynamics of our [Place Holder] training starting from the Qwen-VL-2B, without SFT or reward models. An aha moment and increasing response length is ever observed at a multimodal model.

Similar to DeepSeek R1, self reflection behavior is also observed during our RL training on vision-centric reasoning tasks: the model exhibits an emergent ability to rethink and correct its mistakes.:

```
. . .
Therefore, dark brown wooden bed with white blanket is not above the doorway.
But wait! I can think of something else.
Maybe it's just higher than above the doorway, but slightly lower than above the doorway.
. . .
```

![Reinforcement Learning](https://img.shields.io/badge/Algo-Reinforcement--Learning-red) 
![R1](https://img.shields.io/badge/Algo-R1-red) 
![Vision-Centric](https://img.shields.io/badge/Task-Vision--Perception-yellow) 
![Qwen-VL](https://img.shields.io/badge/Model-Qwen--VL-green)
![Aha-Moment](https://img.shields.io/badge/Analysis-Aha--moment-green) 

**Twitter thread:** [PLACE HOLDER]

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
To reproduce the multimodal aha moment, run the following code:
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


## :coffee: Stay Connected!

We are always open to engaging discussions, collaborations, or even just sharing a virtual coffee. To get in touch or join our team, visit [TurningPoint AI](https://www.turningpoint-ai.com/)'s homepage for contact information.

## 📖 Acknowledgements

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

