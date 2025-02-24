conda create -n TinyZero-V python=3.11 
conda activate TinyZero-V

# Install the packages in open-r1-multimodal .
cd src/open-r1-multimodal # We edit the grpo.py and grpo_trainer.py in open-r1 repo.
pip install -e ".[dev]"

# Addtional modules
pip install wandb==0.18.3
pip install tensorboardx
pip install qwen_vl_utils torchvision
pip install flash-attn --no-build-isolation

pip install transformers==4.49.0 # correct deepspeed support
pip install duckdb
pip install opencv-python
pip install pandas

