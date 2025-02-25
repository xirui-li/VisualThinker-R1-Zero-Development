# Multimodal-R1

This is the under-development repo for tiny-multimodal-r1.

# Setup

```bash
bash setup.sh
```
# Prepare Dataset

```bash
cd src/data/SAT
bash prepare_dataset.sh
```

# Training

## GRPO Training
To reproduce checkpoing on different dataset, replace the sh file in the following command with run_grpo_{dataset_name}.sh
```bash
cd src/open-r1-multimodal
sh run_grpo_SAT.sh # Adjust open-r1-multimodal/configs/zero3.yaml or zero2.yaml accordingly # Full training for 2 epochs take more than 50 hours, we usually can observe reward peak and stop at step 100~500
```

## SFT Training

```bash
cd src/open-r1-multimodal
sh run_sft.sh # Adjust open-r1-multimodal/configs/zero3.yaml or zero2.yaml accordingly
```

# Evaluation

## CVBench Evaluation
```bash
cd src/eval
python test_qwen2vl_CVBench.py 
```

> [!NOTE] 
> 1. To reproduce the result, keep the per_device_train_batch_size to 1 for now, as there is a revealed bug about batched training. See the [reproduction report](https://github.com/Deep-Agent/R1-V/issues/4#issuecomment-2633348354) here. We realize it is important for effiency and are working on solving it with the community.
> 2. If you meet **OOM Error**, add `--deepspeed local_scripts/zero3.json` following https://github.com/Deep-Agent/R1-V/issues/18 or you can reduce `--num_generations`

