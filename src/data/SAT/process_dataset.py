
import os
import io
import re
import json
import random
from PIL import Image
import pandas as pd
import duckdb
# check if the csv file exists
fold = 'train'

if not os.path.exists(f'SAT_{fold}.csv'):
    duckdb.sql(f"""COPY (select * from 'SAT_{fold}.parquet') TO 'SAT_{fold}.csv' (HEADER, FORMAT 'csv')""")

# load csv file
df = pd.read_csv(f'SAT_{fold}.csv')

conversations = []

base_conversations = []

total_num = 15000

# iterate over the dataframe
for index, example in df.iterrows():
    if index > total_num:
        break
    print(index)
    conversation = {}

    images = example['image_bytes'].strip("[]")

    pattern = r"\\xFF\\xD8.*?\\xFF\\xD9"

    # Find all matches
    imags_list = re.findall(pattern, images)

    images = []
    image_token = ""

    single_image = True
    if len(imags_list) > 1:
        continue
    for id, im_bytes in enumerate(imags_list):
        
        no_saving = False
        if not no_saving:

            # import pdb; pdb.set_trace()
            im_bytes = im_bytes.strip().encode().decode('unicode_escape').encode('raw_unicode_escape')
            # save the images to a folder
            image = Image.open(io.BytesIO(im_bytes))
            image.save(f'SAT_images_{fold}/{index}_{id}.png')

        images.append(f'SAT/SAT_images_{fold}/{index}_{id}.png')
        image_token += "<image>"
        
    question = example['question']
    answer_choices = example['answers']
    answer_choices_list = list(answer_choices.strip('[]').split(', '))
    random.shuffle(answer_choices_list)
    correct_answer = example['correct_answer']

    answer = ", ".join(map(str, answer_choices_list[:-1])) + " or " + str(answer_choices_list[-1])
    prompt = f"{question} Choose between the following options: {answer}"
    messages = [{
        "role": "user",
        "content": f"{image_token} Answer in natural language. {prompt}"
    }, {
        "role": "assistant",
        "content": correct_answer
    }]

    conversation["messages"] = messages
    conversation["images"] = images


    question = conversation['messages'][0]["content"]
    base_conversation = conversation.copy()
    base_conversation["messages"] = [{'content': f'A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.\nUser: {question} \nAssistant: Let me solve this step by step.\n<think>', 'role': 'user'}]
    base_conversation["answer"] = conversation['messages'][1]["content"]
    base_conversations.append(base_conversation)

    # save the conversation to a folder
    conversations.append(conversation)

with open(f'SAT_{fold}_{total_num}.json', 'w') as f:
    json.dump(conversations, f, indent=4)

with open(f'SAT_{fold}_{total_num}_base.json', 'w') as f:
    json.dump(base_conversations, f, indent=4)