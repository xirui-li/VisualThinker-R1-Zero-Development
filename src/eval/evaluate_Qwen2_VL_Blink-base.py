import re
import os
import torch
import json
import argparse
import pandas as pd
from tqdm import tqdm
from PIL import Image
from math_verify import parse, verify, LatexExtractionConfig, ExprExtractionConfig, StringExtractionConfig
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from datasets import load_dataset

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True, type=str) # Base model: "Qwen/Qwen2-VL-2B-Instruct"
    parser.add_argument('--bs', default=32, type=int) # Batch size: reduce it if GPU OOM
    parser.add_argument('--output_dir', default="results/Blink", type=str)
    parser.add_argument("--precomputed_json", type=str)
    parser.add_argument("--use_reasoning_prompt", default=True, action=argparse.BooleanOptionalAction)

    return parser.parse_args()

def extract_answer(output_str):
    # Try to find the number within <answer> tags, if can not find, return None
    answer_pattern = r"<answer>\s*(.*?)\s*</answer>"
    match = re.search(answer_pattern, output_str)
    
    if match:
        return match.group(1)
    return None

def extract_characters_regex(s, choices=['(A)', '(B)', '(C)', '(D)', '(E)', '(F)']):
        if type(s) is dict:
            s = ''
        s = s.strip()
        answer_prefixes = [
            'The best answer is',
            'The correct answer is',
            'The answer is',
            'The answer',
            'The best option is'
            'The correct option is',
            'Best answer:'
            'Best option:',
        ]
        for answer_prefix in answer_prefixes:
            s = s.replace(answer_prefix, '')

        if len(s.split()) > 10 and not re.search('[ABCDEF]', s):
            return ''
        matches = re.search(r'[ABCDEF]', s)
        if matches is None:
            for choice in choices:
                if s.lower() in choice.lower():
                    return choice[1]
            return ''
        return matches[0]

def load_images(messsages):
    images = []
    for message in messsages:
        for item in message:
            if item['type'] == 'image':
                if type(item['image']) == str:
                    image_path = item['image']
                    image = Image.open(image_path)
                    images.append(image)
                else:
                    images.append(item['image'])
    return images

if __name__ == "__main__":

    task_list = ['Relative_Depth', 'Relative_Reflectance', 'Spatial_Relation']

    blink_bench = []

    for task in task_list:
        blink = load_dataset("BLINK-Benchmark/BLINK", task, split="val")

        for example in blink:
            if example['image_2'] is None:
                blink_bench.append(example)

    import pdb; pdb.set_trace()

    args = parse_arguments()
    MODEL_PATH=args.model_path
    BSZ=args.bs 
    OUTPUT_DIR=args.output_dir
    PRECOMPUTED_RESULT=args.precomputed_json

    correct_counter = 0
    counter_task = {
        'Relative_Depth': 0,
        'Relative_Reflectance': 0,
        'Spatial_Relation': 0
    }

    counter_correct = {
        'Relative_Depth': 0,
        'Relative_Reflectance': 0,
        'Spatial_Relation': 0
    }
    final_output = []
    if not PRECOMPUTED_RESULT:
        #We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )

        model.eval()
        model = torch.nn.DataParallel(model)

        model.module = torch.compile(model.module)

        # default processer
        processor = AutoProcessor.from_pretrained(MODEL_PATH)
        # processor.tokenizer.padding_side = 'left'

        resp_messages = []

        for i, example in tqdm(enumerate(blink_bench)):

            question = example['prompt']

            if args.use_reasoning_prompt:
                res_prompt = f'A conversation between User and Assistant. The user asks a question about the image, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.\nUser: {question} \nAssistant: Let me solve this step by step.\n<think>'
            else:
                res_prompt = f'A conversation between User and Assistant. The user asks a question about the image, and the Assistant solves it.\nUser: {question} \nAssistant: '

            resp_message = [

                        {
                            "type": "image",
                            "image": example['image'],
                        },
                        {"type": "text", "text": "<image>" + res_prompt},
                    ]

            resp_messages.append(resp_message)

        # List to store all answers
        all_resp_outputs = []

        # Process data in batches
        def generate_batch(batch_messages):

            # Preparation for inference
            text = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batch_messages]
            
            images = load_images(batch_messages)
            inputs = processor(
                text=text,
                images=images,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(model.module.device)
            
            with torch.no_grad():
                generated_ids = model.module.generate(**inputs, use_cache=True, max_new_tokens=1024, do_sample=False, temperature=1)
            
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            batch_output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            return batch_output_text

        for i in tqdm(range(0, len(resp_messages), BSZ)):
            batch_messages = resp_messages[i:i + BSZ]
            
            batch_resp_output = generate_batch(resp_messages[i:i + BSZ])
            
            all_resp_outputs.extend(batch_resp_output)
            print(f"Processed batch {i//BSZ + 1}/{(len(resp_messages) + BSZ - 1)//BSZ}")

    else:
        
        with open(PRECOMPUTED_RESULT, "r") as f:
            result = json.load(f)['results'][:-1]
        all_resp_outputs = [r['response'] for r in result]

    for i, (input_example, model_resp_output) in enumerate(zip(blink_bench, all_resp_outputs)):
        # Count correct answers
        ground_truth = input_example['answer']
        model_answer = extract_answer(model_resp_output)

        if not model_answer:
            short_response = model_resp_output
        else:
            short_response = model_answer

        if input_example['answer'] == '(A)':
            example_answer = input_example['choices'][0]
        elif input_example['answer'] == '(B)':
            example_answer = input_example['choices'][1]
        elif input_example['answer'] == '(C)':
            example_answer = input_example['choices'][2]
        elif input_example['answer'] == '(D)':
            example_answer = input_example['choices'][3]
        elif input_example['answer'] == '(E)':
            example_answer = input_example['choices'][4]
        elif input_example['answer'] == '(F)':
            example_answer = input_example['choices'][5]

        parsed_response = parse(short_response, extraction_config=[LatexExtractionConfig(), ExprExtractionConfig(), StringExtractionConfig(strings=("A", "B", "C", "D", "E", "F", 'a', 'b', 'c', 'd', 'e', 'f', '(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(A)', '(B)', '(C)', '(D)', '(E)', '(F)') + tuple(input_example['choices']))])
        parsed_answer = [example_answer.lower(), example_answer, ground_truth[1], ground_truth[1].lower(), ground_truth.lower(), ground_truth]
        
        if verify(target=parsed_response, gold=parsed_answer):
            correct = 1
            correct_counter += 1
            counter_correct[input_example['task']] +=1
        else:
            correct = 0

        counter_task[input_example['task']] +=1

        result = {
            'question': input_example['question'],
            'options': input_example['choices'],
            "task": input_example['task'],
            'ground_truth': ground_truth,
            'response': model_resp_output,
            "model_answer": short_response,
            "correct": correct,
        }
        final_output.append(result)

    acc = {"Total Accuracy:": correct_counter / len(blink_bench)}
    acc['Relative_Depth'] = counter_correct['Relative_Depth'] / counter_task['Relative_Depth']
    acc['Relative_Reflectance'] = counter_correct['Relative_Reflectance'] / counter_task['Relative_Reflectance']
    acc['Spatial_Relation'] = counter_correct['Spatial_Relation'] / counter_task['Spatial_Relation']

    print(acc)

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    model_name = MODEL_PATH.split('/')[-1]
    reasoning_tag = "reasoning" if args.use_reasoning_prompt else "no_reasoning"
    with open(os.path.join(OUTPUT_DIR, f"CVBench_result_{model_name}_{reasoning_tag}.json"), "w") as f:
        json.dump({
            'results': final_output,
            "accuracy": acc,
            "args": vars(args)
        }, f, indent=2)