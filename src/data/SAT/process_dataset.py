import os
import io
import re
import json
import random
import argparse
from PIL import Image
import pandas as pd
import duckdb
from tqdm import tqdm

def ensure_csv_exists(fold):
    """Ensure the CSV file exists by converting from Parquet if necessary."""
    csv_file = f'SAT_{fold}.csv'
    if not os.path.exists(csv_file):
        duckdb.sql(f"""COPY (SELECT * FROM 'SAT_{fold}.parquet') TO '{csv_file}' (HEADER, FORMAT 'csv')""")
    return csv_file

def extract_images(image_bytes):
    """Extract image bytes from string using regex."""
    pattern = r"\\xFF\\xD8.*?\\xFF\\xD9"
    return re.findall(pattern, image_bytes.strip("[]"))

def save_images(images_list, fold, index):
    """Save images from byte format to PNG files."""
    image_paths = []
    image_folder = f'SAT_images_{fold}'
    os.makedirs(image_folder, exist_ok=True)
    
    for idx, im_bytes in enumerate(images_list):
        im_bytes = im_bytes.strip().encode().decode('unicode_escape').encode('raw_unicode_escape')
        image = Image.open(io.BytesIO(im_bytes))
        image_path = os.path.join(image_folder, f'{index}_{idx}.png')
        image.save(image_path)
        image_paths.append(image_path)
    
    return image_paths

def process_data(df, fold, total_num):
    """Process dataset and generate conversation JSON files."""
    conversations = []
    
    for index, example in tqdm(df.iterrows(), total=total_num, desc="Processing indices"):
        if index >= total_num:
            break
        
        images_list = extract_images(example['image_bytes'])
        if len(images_list) > 1:
            continue  # Skip multiple image cases

        images = save_images(images_list, fold, index)
        image_token = "<image>" if images else ""
        
        question = example['question']
        answer_choices = list(map(str, example['answers'].strip('[]').split(', ')))
        random.shuffle(answer_choices)
        correct_answer = example['correct_answer']
        
        answer = ", ".join(answer_choices[:-1]) + " or " + answer_choices[-1]
        prompt = f"{question} Choose between the following options: {answer}"
        messages = [
            {"role": "user", "content": f"{image_token} Answer in natural language. {prompt}"},
            {"role": "assistant", "content": correct_answer}
        ]
        
        conversation = {"messages": messages, "images": images}

        conversations.append(conversation)
    
    with open(f'SAT_{fold}_{total_num}.json', 'w') as f:
        json.dump(conversations, f, indent=4)


def main():
    parser = argparse.ArgumentParser(description="Process SAT dataset and generate JSON conversations.")
    parser.add_argument('--fold', type=str, default='train', help="Dataset fold to process (e.g., train, val, test)")
    parser.add_argument('--total_num', type=int, default=15000, help="Maximum number of examples to process")
    args = parser.parse_args()
    
    csv_file = ensure_csv_exists(args.fold)
    df = pd.read_csv(csv_file)
    process_data(df, args.fold, args.total_num)

if __name__ == "__main__":
    main()