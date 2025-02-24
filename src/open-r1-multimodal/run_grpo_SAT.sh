export DEBUG_MODE="true" # Enable Debug if you want to see the rollout of model during RL
export LOG_PATH="./debug_log_2b.txt"
export CUDA_VISIBLE_DEVICES=0
export MAIN_PROCESS_PORT=29507  # Change this to an available port

accelerate launch --config_file=configs/zero2.yaml src/open_r1/grpo.py \
    --output_dir outputs/Qwen2-VL-2B-GRPO-SAT-LLM-Freeze \
    --model_name_or_path /nfs/data/xiruili/data_selection/pretrained_models/Qwen2-VL-2B-Instruct \
    --dataset_name SAT \
    --max_prompt_length 512 \
    --max_completion_length 128 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 \
    --gradient_checkpointing false \
    --attn_implementation flash_attention_2 \
    --max_pixels 156800 \
    --num_train_epochs 2 \
    --run_name Qwen2-VL-2B-GRPO-SAT-LLM-Freeze \
    --save_steps 100 \
    --save_only_model true \
    --report_to wandb \
