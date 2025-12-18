# https://platform.openai.com/docs/assistants/overview
import os
import openai
from openai import OpenAI
import argparse
import numpy as np
import importlib
import multiprocessing as mp
import torch.nn as nn

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
    if response == "agree":
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
    model_method = ROMA(config)
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

class Agent:
    def __init__(self, config, role):
        self.config = config
        self.base_role = role
        self.role = role
        questions = np.load(os.path.join(os.path.dirname(__file__)[:-1],'dataset', self.config.dataset, 'question_answer.npy'), allow_pickle=True).item()["questions"][self.base_role]
        answers = np.load(os.path.join(os.path.dirname(__file__)[:-1],'dataset', self.config.dataset, 'question_answer.npy'), allow_pickle=True).item()["answers"][self.base_role]
        self.question_list = questions[:int(0.7*len(questions))]
        self.answer_list = answers[:int(0.7*len(answers))]
    
    def response(self, message):
        messages=[
            {
                "role": "system",
                "content": f"Suppose you are an expert in {self.role}, your job is to answer questions from your expertise."
            },
            {
                "role": "user",
                "content": message
            }
        ]
        r = gen_response(self.config, messages)
        return r

class role_generator(nn.Module):
    def __init__(self, config):
        self.config = config
        self.base_role = config.client_expert
        self.role = self.base_role
    
    def forward(self, agent_list, question):
        pass

    def train(self):
        pass

class ROMA:
    def __init__(self, config):
        self.config = config
        self.agent_list = [Agent(self.config, role) for role in self.config.client_expert]
        self.base_role = self.config.client_expert
        self.role_generator = role_generator(self.config)
    
    def set_role(self, question):
        role_list = role_generator(agent, question)
        for id, agent in enumerate(self.agent_list):
            agent.role = role_list[id]

    def answer_(self, question, id, answer_list):
        messages=f"Please {question}."
        for id, agent in enumerate(self.agent_list):
            response=agent.response(messages)
            answer_list[id]=response

    def answer_debate(self, question, id, answer_list):
        messages=f"Your answer is not good enough. Here are some answers from experts in other fields: {answer_list} Please {question} again from your professional perspective."
        for id, agent in enumerate(self.agent_list):
            response=agent.response(messages)
            answer_list[id]=response
    
    def answer(self, question):
        self.set_role(question)
        response_client_list = []
        for _ in range(self.config.num):
            response_client_list.append("")

        for id in range(self.config.num):
            self.answer_(question, id, response_client_list)
        
        # debate
        for _ in range(self.config.rounds):
            # server evaluate agreement
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
            server_response=gen_response(self.config, messages)
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
        final_answer=gen_response(self.config, messages)
        return final_answer