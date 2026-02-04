import os
import openai
from openai import OpenAI
import argparse
import numpy as np
import importlib
import multiprocessing as mp
import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaTokenizer, T5Tokenizer, T5ForConditionalGeneration
from transformers import LlamaTokenizer, LlamaForCausalLM
import copy
import random
import numpy as np
import subprocess
import itertools
import torch.nn.functional as F
import re
from trl import PPOTrainer, PPOConfig

class Scalar(nn.Module):
    def __init__(self, init_value: float):
        super().__init__()
        self.constant = nn.Parameter(torch.tensor(init_value, dtype=torch.float32))

    def forward(self) -> nn.Parameter:
        return self.constant
    
class ProjectionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4096, 512)
    def forward(self, a):
        return self.fc(a)
    def get_device(self):
        return next(self.parameters()).device

class LightweightF(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(512, 512)
    def forward(self, context):
        return self.fc(context)
    def get_device(self):
        return next(self.parameters()).device

def init_role(role_pool, question, F_model, N, tokenizer, model):
    role_list = []
    M = len(role_pool)
    question_vec = sentence2vec(question, tokenizer, model).sum(dim=0).to(F_model.get_device())
    role_vector = [sentence2vec(role_pool[i]+question, tokenizer, model).to(F_model.get_device()) for i in range(M)]
    best_loss = float("inf")
    best_subset = None

    from itertools import combinations
    for subset in combinations(range(M), N):
        proj = [F_model(role_vector[i].to(F_model.get_device())).sum(dim=0) for i in list(subset)]
        proj = torch.stack(proj)
        omega = torch.linalg.lstsq(proj.T, question_vec).solution
        recon = proj.T @ omega
        loss = torch.norm(recon - question_vec).item()
        if loss < best_loss:
            best_loss = loss
            best_subset = subset
    for i in range(len(subset)):
        role_list.append(role_pool[best_subset[i]])
        
    return role_list

def train_init(role_pool, questions, answers, t5_tokenizer, t5_model, tokenizer, model):
    f_model = ProjectionModel().to('cuda')
    optimzier_f = torch.optim.Adam(f_model.parameters())
    F_model = LightweightF().to('cuda')
    optimzier_F = torch.optim.Adam(F_model.parameters())
    for question, answer in zip(questions, answers):
        v_p = sentence2vec(question, t5_tokenizer, t5_model)
        inputs = tokenizer(answer+question, return_tensors="pt").to('cuda')
        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        a_c = torch.sum(hidden_states[-1].squeeze(dim=0),dim=0)
        f = f_model(a_c).to(v_p.device)
        loss = nn.MSELoss()(v_p.sum(dim=0), f)
        optimzier_f.zero_grad()
        loss.backward()
        optimzier_f.step()
    for question, answer in zip(questions, answers):
        for role in role_pool:
            v_i = sentence2vec(role+question, t5_tokenizer, t5_model).to()
            F_i = F_model(v_i.to(F_model.get_device()))
            inputs = tokenizer(role+question, return_tensors="pt").to('cuda')
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            a_i = torch.sum(hidden_states[-1].squeeze(dim=0),dim=0)
            f_i = f_model(a_i).detach()
            loss = nn.MSELoss()(F_i.sum(dim=0), f_i)
            optimzier_F.zero_grad()
            loss.backward()
            optimzier_F.step()
    return F_model

def sentence2vec(sentence, tokenizer, model):
    device = "cuda"
    inputs = tokenizer(sentence, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        outputs = model.encoder(input_ids=inputs)
        outputs = outputs.last_hidden_state
    sentence_vector = torch.sum(outputs, dim=0)
    return sentence_vector

def verify_answer(config, final_answer, answer):
    pattern = r'\\box(?:ed)?\s*\{([^}]*)\}'
    matches = re.findall(pattern, final_answer)

    if not matches:
        return False

    cleaned_answers = []
    for m in matches:
        m = m.strip()
        if m.startswith('(') and m.endswith(')'):
            m = m[1:-1]
        cleaned_answers.append(m)

    return str(answer) in cleaned_answers

def gen_response(config, tokenizer, model, message, id=1):
    if config.model in ['llama-7b','llama-14b','llama-70b','Qwen-7b','Qwen-13b','Qwen-70b']:
        m = ""
        for i in range(len(message)):
            if id == 1:
                m += message[i]["content"]
                m += " "
        message = m
        inputs = tokenizer(message, return_tensors="pt").to('cuda')
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=config.max_completion_tokens,
                temperature=config.temperature,
                top_k=config.top_k,
                num_return_sequences=1,
            )
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text
    elif config.model in ['qwen-plus']:
        client = OpenAI(
            api_key=config.api_key
        )

        response = client.chat.completions.create(
            model=config.model,
            messages=message
        )
        return response.model_dump_json()
    elif config.model in ['gpt-4o']:
        client = OpenAI(
            api_key=config.api_key
        )

        response = client.chat.completions.create(
            model=config.model,
            messages=message
        )
        return response.choices[0].message.content

def test_(config, F_model, model_method, questions, answers):
    correct = 0
    for question, answer in zip(questions, answers):
        role_list = init_role(config.client_expert, question, F_model, len(model_method.agent_list), model_method.t5_tokenizer, model_method.t5_model)
        for i, agent in enumerate(model_method.agent_list):
            agent.role = role_list[i]
        if(verify_answer(config, model_method.answer(question), answer)):
            correct += 1
    return correct

def test(config, questions, answers):
    model_method = M2CL(config)
    F_model = train_init(config.client_expert, questions, answers, model_method.t5_tokenizer, model_method.t5_model, model_method.tokenizer, model_method.model)
    model_method.train_role(questions, F_model)
    correct = test_(config, F_model, model_method, questions, answers)
    return correct/len(questions)

class Agent:
    def __init__(self, config, role, t5_t, t5_m):
        self.config = config
        self.base_role = role
        self.role = role
        self.t5_tokenizer = t5_t
        self.t5_model = t5_m
        self.role_generator = role_generator(self.config, role, self.t5_tokenizer, self.t5_model)
        self.alpha = Scalar(self.config.alpha_init).to('cuda')
        self.alpha_optimizer = torch.optim.Adam(self.alpha.parameters(), lr=self.config.lr)
    
    def response(self, tokenizer, model, message):
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
        r = gen_response(self.config, tokenizer, model, messages)
        return r
    
    def set_role(self, context):
        self.role, _ = self.role_generator.gen_role(context)

class role_generator(nn.Module):
    def __init__(self, config, role, t5_t, t5_m):
        super(role_generator, self).__init__()
        self.config = config
        self.base_role = role
        self.t5_tokenizer = t5_t
        self.t5_model = t5_m
        self.seq_length = 10
        self.hidden_dim = 512
        self.base_role_vector = sentence2vec(self.base_role, self.t5_tokenizer, self.t5_model)
      
    def gen_role(self, context):
        inputs = self.t5_tokenizer(context, return_tensors="pt").to(self.t5_model.device)
        output = self.t5_model.generate(
            **inputs,
            max_length=self.config.max_completion_tokens,
            do_sample=True,
        )
        role_text = self.t5_tokenizer.decode(
            output[0], skip_special_tokens=True
        )
        return role_text, output

class M2CL:
    def __init__(self, config):
        self.config = config
        model_name_or_path = self.config.model_path
        self.tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path)
        self.model = LlamaForCausalLM.from_pretrained(model_name_or_path, low_cpu_mem_usage=True, device_map="auto")
        model_path = self.config.generator_path
        self.t5_tokenizer = T5Tokenizer.from_pretrained(model_path)
        self.t5_model = T5ForConditionalGeneration.from_pretrained(model_path, device_map="auto")
               
        self.agent_list = [Agent(self.config, "", self.t5_tokenizer, self.t5_model) for i in range(config.num)]
        self.base_role = self.config.client_expert
    
    def gen_reward(self, agent, role, question, response):
        inputs = self.tokenizer(role+question, return_tensors="pt").to('cuda')
        outputs = self.model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        a_i = torch.sum(hidden_states[-1].squeeze(dim=0),dim=0)
        inputs = self.tokenizer(response+question, return_tensors="pt").to('cuda')
        outputs = self.model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        a_x = torch.sum(hidden_states[-1].squeeze(dim=0),dim=0)
        r_1 = torch.sqrt(torch.mean((torch.sum(sentence2vec(role, self.t5_tokenizer, self.t5_model),dim=0) - torch.sum(sentence2vec(agent.base_role, self.t5_tokenizer, self.t5_model),dim=0)) ** 2))
        r_2 = torch.sqrt(torch.mean((a_i- a_x) ** 2))
        reward = r_1.detach()*agent.alpha()+r_2.detach()
        return reward
        
    def sample(self, question):
        context_buffer = [[]]*len(self.agent_list)
        response_buffer = [[]]*len(self.agent_list)
        role_buffer = [[]]*len(self.agent_list)
        reward_buffer = [[]]*len(self.agent_list)
        response_list = ["" for _ in range(len(self.agent_list))]
        for _ in range(self.config.rounds):
            response_list_ = copy.deepcopy(response_list)
            for i in range(len(self.agent_list)):
                responses = ""
                for j in range(len(response_list_)):
                    if j == i:
                        continue
                    responses += response_list_[j]
                role, logits = self.agent_list[i].role_generator.gen_role(question+responses)
                message = role + responses + question
                response_list[i] = gen_response(self.config,self.tokenizer,self.model,message,0)
                response_list[i].replace(message,"")
                reward = self.gen_reward(self.agent_list[i], role, question, response_list_[i])
                context_buffer[i].append(question+responses)
                role_buffer[i].append(role)
                response_buffer[i].append(logits)
                reward_buffer[i].append(reward)
        
        return context_buffer, response_buffer, role_buffer, reward_buffer
    
    def train_alpha(self, id, role_buffer):
        alpha_loss = (self.config.beta - nn.MSELoss()(torch.sum(sentence2vec(role_buffer[id], self.t5_tokenizer, self.t5_model),dim=0).detach(), sentence2vec(self.agent_list[id].base_role, self.t5_tokenizer, self.t5_model).detach()).detach()) * self.agent_list[id].alpha()
        self.agent_list[id].alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.agent_list[id].alpha_optimizer.step()
    
    def train_role(self, questions, F_model):
        ppo_config = PPOConfig(
            model_name='context_generator',
            learning_rate=self.config.lr,
            batch_size=self.config.batch_size,
            mini_batch_size=self.config.batch_size,
            gradient_accumulation_steps=1,
            ppo_epochs=4,
            cliprange=0.2,
            cliprange_value=0.2,
            vf_coef=0.1,
            kl_coef=0.1,
        )
        ppo_trainer_list = [PPOTrainer(
            config=ppo_config,
            model=self.agent_list[i].role_generator.t5_model,
            ref_model=self.t5_model,
            tokenizer=self.t5_tokenizer,
        ) for i in range(len(self.agent_list))]

        for _ in range(self.config.train_rounds):
            query_buffer = [[]]*len(self.agent_list)
            response_buffer = [[]]*len(self.agent_list)
            role_buffer = [[]]*len(self.agent_list)
            reward_buffer = [[]]*len(self.agent_list)
            question_batch = random.sample(questions, self.config.batch_size)
            for question in question_batch:
                role_list = init_role(self.config.client_expert,question,F_model,len(self.agent_list), self.t5_tokenizer, self.t5_model)
                for i, agent in enumerate(self.agent_list):
                    agent.role = role_list[i]
                query, respose, role, reward = self.sample(question)
                for i in range(len(self.agent_list)):
                    query_buffer[i]+=query[i]
                    response_buffer[i]+=respose[i]
                    reward_buffer[i]+=reward[i]
                    role_buffer[i]+=role[i]
                    responses = torch.stack(response_buffer[i], dim=0)

                    ppo_trainer_list[i].step(
                        query_buffer[i],
                        responses,
                        reward_buffer[i],
                    )
                    self.train_alpha(i, role_buffer)
            query_buffer.clear()
            response_buffer.clear()
            reward_buffer.clear()

    
    def set_role(self, question, response_list):
        for id, agent in enumerate(self.agent_list):
            responses = ""
            for i in range(len(response_list)):
                if i == id:
                    continue
                responses += response_list[i]
            agent.set_role(question+responses)

    def answer_(self, question, id, answer_list):
        messages=question
        for id, agent in enumerate(self.agent_list):
            response=agent.response(self.tokenizer, self.model, messages)
            response.replace(messages,"")
            answer_list[id]=response

    def answer_debate(self, question, id, answer_list):
        answers = ""
        for i in range(self.config.num):
            if i == id:
                continue
            else:
                answers += answer_list[i]
        messages=self.agent_list[id].role+answers+question
        response=self.agent_list[id].response(self.tokenizer, self.model, messages)
        response.replace(messages,"")
        
        return response
    
    def answer(self, question):
        response_client_list = []
        for _ in range(self.config.num):
            response_client_list.append("")

        for id in range(self.config.num):
            self.answer_(question, id, response_client_list)
        
        for _ in range(self.config.rounds-1):
            self.set_role(question, response_client_list)
            temp = copy.deepcopy(response_client_list)
            for id in range(self.config.num):
                response = self.answer_debate(question, id, temp)
                response_client_list[id] = response
        
        # final answer
        messages=[
            {
                "role": "system",
                "content": f"You are a leader and summarizer. Your job is to assess how well your group answers {question}."
            },
            {
                "role": "user",
                "content": f"Please go through the following responses {response_client_list} Then, summarize your final answer to the question {question}. "
            }
        ]
        final_answer=gen_response(self.config, self.tokenizer, self.model, messages)
        final_answer.replace(question,"")
        return final_answer