<div align="center">

# VisualThinker R1 Zero

[![Notion](https://img.shields.io/badge/Notion-%23000000.svg?style=for-the-badge&logo=notion&logoColor=white)](https://turningpointai.notion.site/the-multimodal-aha-moment-on-2b-model)

![Reinforcement Learning](https://img.shields.io/badge/Algo-Reinforcement--Learning-red) 
![R1](https://img.shields.io/badge/Algo-R1-red) 
![Vision-Centric](https://img.shields.io/badge/Task-Vision--Centric-yellow) 
![Qwen2-VL-2B](https://img.shields.io/badge/Model-Qwen2--VL--2B-green)
![Aha-Moment](https://img.shields.io/badge/Analysis-Aha--moment-blue) 

</div>

<div align="center">
<img src="https://multimodal-r1.s3.us-west-1.amazonaws.com/Training_Steps.png" width="700" alt="visualthinking-intro-figure_00">
</div>

> Training dynamics of our VisionThinker-R1-Zero training starting from the Qwen-VL-2B, without SFT or reward models. An aha moment and increasing response length is ever observed at a multimodal model.

[DeepSeek R1](https://arxiv.org/abs/2501.12948) has demonstrated how Reinforcement Learning (RL) with well-designed rule-based rewards can enable a large language model to build unique reasoning capabilities autonomously. Since then, many researchers have attempted to extend this success to multimodal reasoning. However, recent efforts primarily **struggle to reproduce the increasing response length and thinking pattern** exhibited by DeepSeek R1. 

VisionThinker-R1-Zero is a replication of DeepSeek-R1-Zero training on **small multimodal** models. We are **the first** to successfully observe **the emergent “aha moment”** and **increased response** length on **multimodal** tasks.
Through applying GRPO on the 2B base LM develops self-verification autonomously and exhibits an emergent ability to "take another look" at the image and correct its mistakes.

1. We are the **first to replicate the key characteristics** of R1 success (**”aha moment”** and **increasing reasoning length**) on **multimodal** reasoning tasks.

2. We showed that **vision-centric** tasks could also benefit from improved reasoning capabilities.  

Similar to DeepSeek R1, self reflection behavior is also observed during our RL training on vision-centric reasoning tasks. The model exhibits an emergent ability to rethink and correct its mistakes:

```
. . .
Therefore, dark brown wooden bed with white blanket is not above the doorway.
But wait! I can think of something else.
Maybe it's just higher than above the doorway, but slightly lower than above the doorway.
. . .
```

**Full experiment log:** Upcoming

**Models CKPT:** Upcoming

### Updates:
- 2025-02-26: We share our main findings in this [notion blog](https://turningpointai.notion.site/the-multimodal-aha-moment-on-2b-model).
- 2025-02-26: We release the VisualThinker R1 Zero repo.

### Teams:
[Hengguang Zhou](https://hengguangzhou.github.io/), [Xirui Li](https://xirui-li.github.io/), [Ruochen Wang](https://ruocwang.github.io/), [Minhao Cheng](https://cmhcbb.github.io/), [Tianyi Zhou](https://tianyizhou.github.io/) and [Cho-Jui Hsieh](https://web.cs.ucla.edu/~chohsieh/)

<div align="center">
<img src="https://multimodal-r1.s3.us-west-1.amazonaws.com/TurningPoint_transparent.png" width="300" alt="logo-figure_00">
<img src="https://multimodal-r1.s3.us-west-1.amazonaws.com/logo.png" width="700" alt="logo-figure_00">
</div>

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
To reproduce the multimodal aha moment, run the following code to train the unaligned base model with GRPO on SAT:
```bash
cd src/open-r1-multimodal
bash run_grpo_SAT.sh # Adjust open-r1-multimodal/configs/zero3.yaml or zero2.yaml accordingly
```

### SFT Training
To obtain SFT model for comparison, run the following code to train the unaligned base model on SAT:
```bash
cd src/open-r1-multimodal
bash run_sft.sh # Adjust open-r1-multimodal/configs/zero3.yaml or zero2.yaml accordingly
```

## Evaluation

### CVBench Evaluation
First change to evaluation directory:
```bash
cd src/eval 
```

To evaluate Base + GRPO (VisualThinker R1 Zero) model:
```bash
python evaluate_Qwen2_VL_CVBench-base.py --model_path <path_to_your_model> \
    --bs 8 \
    --use_reasoning_prompt
```
To evaluate Base model:
```bash
python evaluate_Qwen2_VL_CVBench-base.py --model_path <path_to_your_model> \
    --bs 8 \
    --no-use_reasoning_prompt
```
To evaluate Instruct + GRPO model:
```bash
python evaluate_Qwen2_VL_CVBench.py --model_path <path_to_your_model> \
    --bs 8 \
    --use_reasoning_prompt
```
To evaluate Instruct model:
```bash
python evaluate_Qwen2_VL_CVBench.py --model_path <path_to_your_model> \
    --bs 8 \
    --no-use_reasoning_prompt
```


## :coffee: Stay Connected!

We are always open to engaging discussions, collaborations, or even just sharing a virtual coffee. To get in touch or join our team, visit [TurningPoint AI](https://www.turningpoint-ai.com/)'s homepage for contact information.

## 📖 Acknowledgements

We sincerely thank [DeepSeek](https://github.com/deepseek-ai/DeepSeek-R1), [Open-R1](https://github.com/huggingface/open-r1), [QwenVL](https://github.com/QwenLM/Qwen2.5-VL), [Open-R1-Multimodal](https://github.com/EvolvingLMMs-Lab/open-r1-multimodal), [R1-V](https://github.com/Deep-Agent/R1-V), [SAT](https://arxiv.org/abs/2412.07755), and [CV-Bench](https://cambrian-mllm.github.io/) for providing open source resources that laid the foundation of our project. 

## :white_check_mark: Cite

If you find our work useful for your projects, please kindly cite the following BibTeX:

```latex
@misc{zhou2024visualthinkerr1,
      title={The Multimodal “Aha Moment” on 2B Model}, 
      author={Hengguang Zhou and Xirui Li and Ruochen Wang and Minhao Cheng and Tianyi Zhou and Cho-Jui Hsieh},
      journal= {arXiv preprint arXiv:XXXX.XXXXX},
      year={2025},
      url={https://arxiv.org/abs/XXXX.XXXXX}
}
```

