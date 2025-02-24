export CUDA_VISIBLE_DEVICES=0

accelerate launch --config_file=configs/zero3.yaml src/open_r1/sft_qwen_vl.py \
    --model_name_or_path Qwen/Qwen2-VL-2B \
    --dataset_name SAT \
    --learning_rate 2.0e-5 \
    --num_train_epochs 2 \
    --packing True \
    --max_seq_length 2048 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --gradient_checkpointing True \
    --report_to wandb \
    --bf16 True \
    --logging_steps 5 \
    --eval_strategy no \
    --output_dir data/Qwen2_VL-2B-SFT \
    --run_name Qwen2_VL-2B-SFT-SAT