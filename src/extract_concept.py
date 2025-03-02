import os
import json
import glob
import matplotlib.pyplot as plt
from openai import OpenAI
import yaml
import tqdm
from math import ceil
import argparse
import concurrent.futures

with open('prompt.txt', 'r') as f:
    prompt = f.read()



def get_concpet(sentence):
  response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
      {
        "role": "system",
        "content": [
          {
            "type": "text",
            "text": prompt
          }
        ]
      },
      {
        "role": "user",
        "content": [
          {
            "text": sentence,
            "type": "text"
          }
        ]
      }
      
    ],
    response_format={
      "type": "text"
    },
    temperature=1,
    max_tokens=16383,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
  )
  return response



def process_key(key, finance, chatgpt4_mini_think, error_think):
    if key in chatgpt4_mini_think.keys():
        return None, None

    article = finance[key]
    if len(article) == 0:
        return None, None

    #print(key, len(article))
    runs = ceil(len(article) / 3000)
    local_chatgpt4_mini_think = {key: {}}
    local_error_think = {key: {}}

    for i in range(runs):
        start = i * 3000
        end = (i + 1) * 3000
        if end > len(article):
            end = len(article)

        test = get_concpet(article[start:end])
        try:
            local_chatgpt4_mini_think[key][int(i)] = json.loads(test.choices[0].message.content)
        except:
            local_error_think[key][int(i)] = test.choices

    return local_chatgpt4_mini_think, local_error_think




if __name__ == "__main__":
    with open('config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    client = OpenAI(api_key=config['api_key'])
    
    if os.path.exists('chatgpt4_mini_think.json'):
        with open('chatgpt4_mini_think.json', 'r') as f:
            chatgpt4_mini_think = json.load(f)
    else:
        chatgpt4_mini_think = {}


    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(process_key, key, finance, chatgpt4_mini_think, error_think): key for key in finance.keys()}

        for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            local_chatgpt4_mini_think, local_error_think = future.result()
            if local_chatgpt4_mini_think:
                chatgpt4_mini_think.update(local_chatgpt4_mini_think)


                # Save to JSON files after each key is processed
                with open('chatgpt4_mini_think.json', 'w') as f:
                    json.dump(chatgpt4_mini_think, f, ensure_ascii=False)
