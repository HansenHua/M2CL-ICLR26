# https://platform.openai.com/docs/assistants/overview
import os
import openai
from openai import OpenAI
import argparse
import numpy as np
import importlib
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
    model_method = Debate(config)
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

def check_agreement(response):
    response_list = response.split(' ')
    if 'agree' in response_list:
        return True
    else:
        return False

class Debate:
    def __init__(self, config):
        self.config = config
        self.server = OpenAI(api_key=config.API_KEY)
        self.client_list = []
        for _ in range(self.config.num):
            self.client_list.append(OpenAI(api_key=config.API_KEY))

    def answer_(self, question, id, answer_list):
        messages=[
            {
                "role": "user",
                "content": f"Please {question}."
            }
        ]
        response = gen_response(self.config, messages)
        # response = self.client_list[id].chat.completions.create(
        #         model=self.config.model,
        #         messages=[
        #             {
        #                 "role": "user",
        #                 "content": f"Please {question}."
        #             }
        #         ],
        #         temperature=self.config.temperature,
        #         max_completion_tokens=self.config.max_completion_tokens,
        #         seed=self.config.seed
        #     )
        answer_list[id]=response

    def answer_debate(self, question, id, answer_list):
        messages=[
            {
                "role": "user",
                "content": f"Your answer is not good enough. Here are some answers from experts in other fields: {answer_list} Please {question} again from your professional perspective."
            }
        ]
        response = gen_response(self.config, messages)
        # response = self.client_list[id].chat.completions.create(
        #         model=self.config.model,
        #         messages=[
        #             {
        #                     "role": "user",
        #                     "content": f"Your answer is not good enough. Here are some answers from experts in other fields: {answer_list} Please {question} again from your professional perspective."
        #             }
        #         ],
        #         temperature=self.config.temperature,
        #         max_completion_tokens=self.config.max_completion_tokens,
        #         seed=self.config.seed
        #     )
        answer_list[id]=response
    
    def answer(self, question):
        response_client_list = []
        for _ in range(self.config.num):
            response_client_list.append("")

        for id in range(self.config.num):
            self.answer_(question, id, response_client_list)
        
        # debate
        for _ in range(self.config.rounds):
            messages=[
                {
                    "role": "system",
                    "content": f"You are a leader and summarizer. Your job is to assess how well your group answers {question} and rank them."
                },
                {
                    "role": "user",
                    "content": f"Do you think the following answers have reached an agreeent? please answer with agree or disagree. {response_client_list}"
                }
            ]
            server_response = gen_response(self.config, messages)
            # server_response = self.server.chat.completions.create(
            #     model=self.config.model,
            #     messages=[
            #         {
            #             "role": "system",
            #             "content": f"You are a leader and summarizer. Your job is to assess how well your group answers {question} and rank them."
            #         },
            #         {
            #             "role": "user",
            #             "content": f"Do you think the following answers have reached an agreeent? please answer with agree or disagree. {response_client_list}"
            #         }
            #     ],
            #     temperature=self.config.temperature,
            #     max_completion_tokens=self.config.max_completion_tokens,
            #     seed=self.config.seed
            # )
            if(check_agreement(server_response)):
                break

            for id in range(self.config.num):
                self.answer_debate(question, id, response_client_list)
        
        # final answer
        messages=[
            {
                "role": "system",
                "content": f"You are a leader and summarizer. Your job is to assess how well your group answers {question} and rank them."
            },
            {
                "role": "user",
                "content": f"Please go through the following responses and summarize your final answer. {response_client_list}"
            }
        ]
        final_answer = gen_response(self.config, messages)
        # final_answer = self.server.chat.completions.create(
        #     model=self.config.model,
        #     messages=[
        #         {
        #             "role": "system",
        #             "content": f"You are a leader and summarizer. Your job is to assess how well your group answers {question} and rank them."
        #         },
        #         {
        #             "role": "user",
        #             "content": f"Please go through the following responses and summarize your final answer. {response_client_list}"
        #         }
        #     ],
        #     temperature=self.config.temperature,
        #     max_completion_tokens=self.config.max_completion_tokens,
        #     seed=self.config.seed
        # )
        return final_answer