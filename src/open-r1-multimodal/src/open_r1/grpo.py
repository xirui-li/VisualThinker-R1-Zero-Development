# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional
import cv2
import numpy as np

from datasets import load_dataset, load_from_disk, concatenate_datasets
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

from math_verify import parse, verify
from open_r1.trainer import Qwen2VLGRPOTrainer
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config
from PIL import Image

@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )


def extract_letters(text): # for RAVEN
    pattern = r'(^|\s|\[|\()([A-H])(\s|\]|\)|$)'
    matches = re.findall(pattern, text)
    return [match[1] for match in matches]

def accuracy_reward(completions, solution, **kwargs):
    """Reward function that checks if the completion is correct using either symbolic verification or exact string matching."""
    if isinstance(completions[0],str):
        contents = [completion for completion in completions]
    else:
        contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    for content, sol in zip(contents, solution):
        reward = 0.0
        # Try symbolic verification first
        try:
            answer = parse(content)
            if float(verify(answer, parse(sol))) > 0:
                reward = 1.0
        except Exception:
            pass  # Continue to next verification method if this fails

        # If symbolic verification failed, try string matching
        if reward == 0.0:
            try:
                # Extract answer from solution if it has think/answer tags
                sol_match = re.search(r'<answer>(.*?)</answer>', sol)
                ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()
                
                # Extract answer from content if it has think/answer tags
                content_match = re.search(r'<answer>(.*?)</answer>', content)
                student_answer = content_match.group(1).strip() if content_match else content.strip()
                
                if extract_letters(student_answer)[-1] == ground_truth:
                    reward = 1.0
                # Compare the extracted answers
                if student_answer == ground_truth:
                    reward = 1.0
            except Exception:
                pass  # Keep reward as 0.0 if both methods fail
                
        rewards.append(reward)
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            # local_rank = int(os.getenv("LOCAL_RANK", 0))
            with open(log_path, "a") as f:
                f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                f.write(f"Content: {content}\n")
                f.write(f"Solution: {sol}\n")
    return rewards

def length_reward(completions, **kwargs):
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
    rewards = []
    for completion in completions:
        rewards.append(len(processor.tokenizer(completion[0]["content"])['input_ids']) * 0.001)
    return rewards

def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    if isinstance(completions[0],str):
        completion_contents = ["<think>" + completion for completion in completions]
    else:
        completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]


reward_funcs_registry = {
    "accuracy": accuracy_reward,
    "format": format_reward,
    "length": length_reward,
}

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)

PROMPT_LIST = [
    "How many objects are present in the image?",
    "What is the total count of items in the image?",
    "How many elements can be seen in the image?",
    "What is the number of items depicted in the image?",
    "How many things are shown in the image?",
    "Can you count the objects in the image?",
    "What is the total number of objects in the picture?",
    "How many elements are visible in the image?",
    "What is the object count in the image?",
    "How many entities are present in the image?",
    "What's the total quantity of items in the image?",
    "Can you determine the number of items in the image?",
    "How many distinct objects are in the image?",
    "What's the total number of things in the picture?",
    "How many components are in the image?",
    "What is the sum of all items in the image?",
    "How many figures can be seen in the image?",
    "What's the count of elements in the picture?",
    "How many objects can be identified in the image?",
    "Can you quantify the number of items in the image?"
]
import random

# RAVEN Processing utilities
BORDER = 60
IMAGE_SIZE = 160
def generate_matrix(array_list, return_subfigure=False):
    # row-major array_list
    assert len(array_list) <= 9
    img_grid = np.zeros(((IMAGE_SIZE+BORDER) * 3 , (IMAGE_SIZE+BORDER) * 3 ), np.uint8)
    subfigures = []
    for idx in range(len(array_list) + 1):
        i, j = divmod(idx, 3)
        
        if idx == 8:
            background = np.ones((IMAGE_SIZE+BORDER, IMAGE_SIZE+BORDER), np.uint8) * 255
            cv2.putText(background, "?", ((IMAGE_SIZE+BORDER) // 2, (IMAGE_SIZE+BORDER) // 2), cv2.FONT_HERSHEY_SIMPLEX, 2, 0, 1)
        else:
            subfigure = array_list[idx]
            # zoom in
            background = np.ones((IMAGE_SIZE+BORDER, IMAGE_SIZE+BORDER), np.uint8) * 255
            # subfigure = cv2.resize(subfigure, (IMAGE_SIZE//2, IMAGE_SIZE//2), interpolation=cv2.INTER_NEAREST)
            # apply subfigure on top of background
            background[BORDER//2:BORDER//2+subfigure.shape[0], BORDER//2:BORDER//2+subfigure.shape[1]] = subfigure
        img_grid[i * (IMAGE_SIZE+BORDER):(i + 1) * (IMAGE_SIZE+BORDER), j * (IMAGE_SIZE+BORDER):(j + 1) * (IMAGE_SIZE+BORDER)] = background
        subfigures.append(background.copy())
    # draw grid
    for x in [0.33, 0.67]:
        img_grid[int(x * (IMAGE_SIZE+BORDER) * 3) - 1:int(x * (IMAGE_SIZE+BORDER) * 3) + 1, :] = 0
    for y in [0.33, 0.67]:
        img_grid[:, int(y * (IMAGE_SIZE+BORDER) * 3) - 1:int(y * (IMAGE_SIZE+BORDER) * 3) + 1] = 0
    if return_subfigure:
        return subfigures
    else:
        return img_grid

def generate_answers(array_list, return_subfigure=False):
    assert len(array_list) <= 8
    img_grid = np.zeros(((IMAGE_SIZE+BORDER) * 2, (IMAGE_SIZE+BORDER) * 4), np.uint8)
    map_id2letter = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5:"F", 6:"G", 7:"H"}
    subfigures = []
    for idx in range(len(array_list)):
        i, j = divmod(idx, 4)
        subfigure = array_list[idx]
        # zoom in
        background = np.ones(((IMAGE_SIZE+BORDER), (IMAGE_SIZE+BORDER)), np.uint8) * 255
        # subfigure = cv2.resize(subfigure, ((IMAGE_SIZE)//2, (IMAGE_SIZE)//2), interpolation=cv2.INTER_NEAREST)
        # apply subfigure on top of background
        background[int((BORDER)//2):int((BORDER)//2)+subfigure.shape[0], int((BORDER)//2):int((BORDER)//2)+subfigure.shape[1]] = subfigure
        # add text to the top left corner
        title = map_id2letter[idx]
        
        font_size = 0.6
        cv2.putText(background, title, (int((IMAGE_SIZE+BORDER)//8), int((IMAGE_SIZE+BORDER)//8)), cv2.FONT_HERSHEY_SIMPLEX, font_size, 0, 1)
        img_grid[i * (IMAGE_SIZE+BORDER):(i + 1) * (IMAGE_SIZE+BORDER), j * (IMAGE_SIZE+BORDER):(j + 1) * (IMAGE_SIZE+BORDER)] = background
        subfigures.append(background.copy())
    # draw grid
    for x in [0.5]:
        img_grid[int(x * (IMAGE_SIZE+BORDER) * 2) - 1:int(x * (IMAGE_SIZE+BORDER) * 2) + 1, :] = 0
    for y in [0.25, 0.5, 0.75]:
        img_grid[:, int(y * (IMAGE_SIZE+BORDER) * 4) - 1:int(y * (IMAGE_SIZE+BORDER) * 4) + 1] = 0
    if return_subfigure:
        return subfigures
    else:
        return img_grid
    
def generate_matrix_answer(array_list):
    # row-major array_list
    assert len(array_list) <= 18
    img_grid = np.zeros((IMAGE_SIZE * 6, IMAGE_SIZE * 3), np.uint8)
    for idx in range(len(array_list)):
        i, j = divmod(idx, 3)
        img_grid[i * IMAGE_SIZE:(i + 1) * IMAGE_SIZE, j * IMAGE_SIZE:(j + 1) * IMAGE_SIZE] = array_list[idx]
    # draw grid
    for x in [0.33, 0.67, 1.00, 1.33, 1.67]:
        img_grid[int(x * IMAGE_SIZE * 3), :] = 0
    for y in [0.33, 0.67]:
        img_grid[:, int(y * IMAGE_SIZE * 3)] = 0
    return img_grid

def merge_matrix_answer(matrix, answer):
    matrix_image = generate_matrix(matrix)
    answer_image = generate_answers(answer)
    img_grid = np.ones(((IMAGE_SIZE+BORDER) * 5 + 20, (IMAGE_SIZE+BORDER) * 4), np.uint8) * 255
    img_grid[:(IMAGE_SIZE+BORDER) * 3, int(0.5 * (IMAGE_SIZE+BORDER)):int(3.5 * (IMAGE_SIZE+BORDER))] = matrix_image
    img_grid[-((IMAGE_SIZE+BORDER) * 2):, :] = answer_image
    return img_grid

def process_segments(images, choices):
    images = [np.array(img) for img in images]
    choices = [np.array(img) for img in choices]
    final_image = merge_matrix_answer(images, choices)

    return final_image

def main(script_args, training_args, model_args):
    # Get reward functions
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]

    # Check if the model is a base model
    base_model_prompt = False
    if model_args.model_name_or_path.split("/")[-1] == "Qwen2-VL-2B" or "Base" in model_args.model_name_or_path:
        base_model_prompt = True

    # Format into conversation
    def make_conversation(example):
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["problem"]},
            ],
        }
    
    QUESTION_TEMPLATE = "{Question}  Output the thinking process in <think> </think> and final answer (option) in <answer> </answer> tags."
    
    if script_args.dataset_name == "RAVEN":
        RAVEN_prompt = "The image displays an intelligence test question featuring a 3x3 grid with nine boxes, where the 9th box is marked with a question mark (?). Your task is to select the correct shape from eight options (labeled A to H) to fill the 9th box, completing the pattern that links all the shapes together."
        map_id2letter = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5:"F", 6:"G", 7:"H"}
        def make_conversation_RAVEN(example):
            images = example['panels']
            choices = example['choices']
            final_image = Image.fromarray(process_segments(images, choices))
            prompt = f'A conversation between User and Assistant. The user asks a question about the image, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.\nUser: {RAVEN_prompt} \nAssistant: Let me solve this step by step.\n<think>'
            if base_model_prompt:
                return {"image": final_image,
                    "prompt": [
                                {"type": "image"},
                                {"type": "text", "text": "<image>" + prompt}],
                    "solution":  "<answer>" + map_id2letter[example['target']] + "</answer>", 
                }
            else:
                return {
                    "image": final_image,
                    "prompt": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image"},
                                {"type": "text", "text": QUESTION_TEMPLATE.format(Question=RAVEN_prompt)},
                            ],
                        },
                    ],
                    "solution":  "<answer>" + map_id2letter[example['target']] + "</answer>", 
                }
        dataset = load_dataset("HuggingFaceM4/RAVEN", "center_single")
        dataset = dataset['train'].map(make_conversation_RAVEN)
        dataset = {'train': dataset}
    elif script_args.dataset_name == "vision-centric":
        short_answer_prompt1 = "Please answer directly with only the letter of the correct option and nothing else."
        short_answer_prompt2 = "Please answer directly with a single word or number."
        def make_conversation_realworldqa(example):
            image = example['image']
            return {"image": image,
                "prompt": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": QUESTION_TEMPLATE.format(Question=example['question'].replace(short_answer_prompt1, '').replace(short_answer_prompt2, ''))},
                        ],
                    },
                ],
                "solution":  "<answer>" + example['answer'] + "</answer>", 
            }

        def make_conversation_tallyqa(example):
            return {
                "image": example['image'],
                "prompt": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": QUESTION_TEMPLATE.format(Question=example['question'].replace(short_answer_prompt1, '').replace(short_answer_prompt2, ''))},
                        ],
                    },
                ],
                "solution":  "<answer>" + example['answer'] + "</answer>", 
            }
        realworld_qa = load_dataset("xai-org/RealworldQA")
        realworld_qa = realworld_qa.map(make_conversation_realworldqa, remove_columns=["answer"])
        tally_qa = load_dataset("vikhyatk/tallyqa-test")

        NUM_TALLY = 700

        def filter_qa_pairs(example):
            """
            Filters out QA pairs where `is_simple == True`, keeping only complex ones.
            """
            example["qa"] = [qa for qa in example["qa"] if not qa["is_simple"]]
            return example
        
        def flatten_tally(batch):
            """
            Takes a dataset row with an image and a list of QA pairs, 
            and returns a flattened version where each QA pair is a separate row.
            """
            new_examples = {"image": [], "question": [], "prompt": [], "solution": []}

            for image, qa_pairs in zip(batch["image"], batch["qa"]):
                for qa in qa_pairs:
                    new_examples["image"].append(image)  # Keep the image the same
                    new_examples["question"].append(qa["question"])
                    new_examples["solution"].append(qa["answer"])
                    new_examples['prompt'].append([
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": QUESTION_TEMPLATE.format(Question=qa['question'])},
                        ],
                    },
                ])
                    # new_examples["is_simple"].append(qa["is_simple"])
                    # new_examples["data_source"].append(qa["data_source"])
            
            return new_examples

        tally_qa = tally_qa['test'].map(filter_qa_pairs)

        filtered_tally = tally_qa.filter(lambda x: len(x["qa"]) > 0)
        flattened_tally = filtered_tally.map(flatten_tally, batched=True, remove_columns=["qa"])

        final_tally = flattened_tally.shuffle(seed=42).select(range(NUM_TALLY))
        dataset = concatenate_datasets([realworld_qa['test'], final_tally])
        dataset = {'train': dataset}
    elif script_args.dataset_name == "SAT":
        def make_conversation_sat(example, base_model_prompt=False):
            if "paligemma" in model_args.model_name_or_path.lower():
                prompt = f'A conversation between User and Assistant. The user asks a question about the image, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>.\n User:{question}'
                return {"image": image,
                        "prompt": "<image>" + prompt + "Assistant: Let me solve this step by step. \n<think>",
                    "solution":  "<answer>" + example["messages"][1]["content"] + "</answer>", 
                }
            else:
                if base_model_prompt:
                    image = Image.open(dataset_prefix + example["images"][0])
                    question = example["messages"][0]["content"]
                    question = question.replace("<image>", "")
                    prompt = f'A conversation between User and Assistant. The user asks a question about the image, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.\nUser: {question} \nAssistant: Let me solve this step by step.\n<think>'
 
                    return {"image": image,
                        "prompt": [
                                    {"type": "image"},
                                    {"type": "text", "text": "<image>" + prompt}],
                        "solution":  "<answer>" + example["messages"][1]["content"] + "</answer>", 
                    }
                else:
                    image = Image.open(dataset_prefix + example["images"][0])
                    return {"image": image,
                        "image_path": example["images"][0],
                        "prompt": [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image"},
                                    {"type": "text", "text": QUESTION_TEMPLATE.format(Question=example["messages"][0]["content"])},
                                ],
                            },
                        ],
                        "solution":  "<answer>" + example["messages"][1]["content"] + "</answer>", 
                    }
        dataset_prefix = "../data/SAT/"
        dataset_path = "SAT_train_15000.json"
        
        import json
        # load json file 
        with open(dataset_prefix + dataset_path, 'r') as f:
            sat_dataset = json.load(f)

        dataset = [make_conversation_sat(sample, base_model_prompt) for sample in sat_dataset]
        dataset = {'train': dataset}

    elif script_args.dataset_name == "LLaVA-OneVision":
        data_list = [
                     'CLEVR-Math(MathV360K)', 
                     'GEOS(MathV360K)', 
                     'Geometry3K(MathV360K)', 
                     'UniGeo(MathV360K)', 
                     'TabMWP(MathV360K)', 
                     'FigureQA(MathV360K)'
                     ]
        
        dataset_list = []
        data_len = []
        def make_conversation_image(example):
            return {
                "prompt": [
                    {"role": "user", "content": [{"type": "image"}, 
                                                {"type": "text", "text": QUESTION_TEMPLATE.format(Question=example["conversations"][0]["value"])}
                                                ]
                    },
                ],
                "solution": example["conversations"][1]["value"],
                "image": example['image']
            }

        for data_name in data_list:
            data = load_dataset("lmms-lab/LLaVA-OneVision-Data", data_name, split="train")
            dataset_list.append(data)
            data_len.append(len(data))
        dataset = []
        # random sample dataset with ratio 0.1
        ratio = 0.1
        for i in range(len(dataset_list)):
            dataset_list[i] = dataset_list[i].shuffle(seed=42).select(range(int(data_len[i]*ratio)))
            dataset += [make_conversation_image(sample) for sample in dataset_list[i]]
        
        dataset = {'train': dataset}
    else:
        # Load the dataset
        dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)


        # Format into conversation
        def make_conversation(example):
            return {
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": example["problem"]},
                ],
            }
        def make_conversation_image(example):
            return {
                "prompt": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": QUESTION_TEMPLATE.format(Question=example["problem"])},
                        ],
                    },
                ],
            }
        
        if "image" in dataset[script_args.dataset_train_split].features:
            print("has image in dataset")
            dataset = dataset.map(make_conversation_image)  # Utilize multiprocessing for faster mapping
            # dataset = dataset.remove_columns(["original_question", "original_answer"])

        else:
            print("no image in dataset")
            dataset = dataset.map(make_conversation)
            dataset = dataset.remove_columns("messages")

    trainer_cls = Qwen2VLGRPOTrainer

    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
    )

    # trainer.model.visual.requires_grad_ = False # Uncomment for vision-encoder freezed training
    # Train and push the model to the Hub
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
