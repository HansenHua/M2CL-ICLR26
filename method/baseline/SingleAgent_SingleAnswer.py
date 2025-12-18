import os
import openai
from openai import OpenAI
import numpy as np
import multiprocessing as mp

def verify_answer(config, final_answer, answer):
    messages=[
        {
            "role": "system",
            "content": f"Your job is to verify if the two answer are the same. You can only say 'yes' or 'no'."
        },
        {
            "role": "user",
            "content": f"Please verify if the following answer is correct: {final_answer} {answer}"
        }
    ]
    response = gen_response(config, messages)
    if response == "yes":
        return True
    else:
        return False

def gen_response(config, message):
    import http.client
    import json

    conn = http.client.HTTPSConnection("api.chatanywhere.tech")
    payload = json.dumps({
    "model": config.model,
    "messages": message,
    "temperature":config.temperature,
    "max_tokens":config.max_completion_tokens,
    })
    headers = {
    'Authorization': config.api_key,
    'Content-Type': 'application/json'
    }
    conn.request("POST", "/v1/chat/completions", payload, headers)
    res = conn.getresponse()
    data = res.read()
    data = data.decode("utf-8")
    data = json.loads(data)
    if "choices" not in data:
        print("error generating answer")
        return data
    return data["choices"][0]["message"]["content"]

def test_(config, questions, answers, id, correct_list):
    model_method = SingleAgent_SingleAnswer(config)
    correct = 0
    for question, answer in zip(questions, answers):
        if(verify_answer(config, model_method.answer(question), answer)):
            correct += 1
    correct_list[id] = correct
    return correct

def test(config, questions, answers):
    p_list = []
    correct_list = [0 for i in range(config.num)]
    for id in range(config.num):
        p_list.append(mp.Process(target=test_, args=(config, questions, answers, id, correct_list,)))
    [p.start() for p in p_list]
    [p.join() for p in p_list]
    return sum(correct_list)/len(questions)

class SingleAgent_SingleAnswer:
    def __init__(self, config):
        self.config = config
    
    def answer(self, question):
        messages=[
            {
                "role": "user",
                "content": f"Please {question} from your professional perspective."
            }
        ]
        final_answer = gen_response(self.config, messages)
        # final_answer = self.agent.chat.completions.create(
        #     model=self.config.model,
        #     messages=[
        #         {
        #             "role": "user",
        #             "content": f"Please {question} from your professional perspective."
        #         }
        #     ],
        #     temperature=self.config.temperature,
        #     max_completion_tokens=self.config.max_completion_tokens,
        #     seed=self.config.seed,
        # )
        
        return final_answer