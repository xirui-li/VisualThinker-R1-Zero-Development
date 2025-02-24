# Download the dataset parquet and rename it
wget -O SAT_train.parquet "https://huggingface.co/datasets/array/SAT/resolve/main/SAT_train.parquet?download=true"

# Create the dataset directory
mkdir -p SAT_images_train

# Process the dataset
python process_dataset.py