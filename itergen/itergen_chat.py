import argparse
import re
from time import sleep

import requests
import json
import os
import concurrent.futures

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

params_baichuan = {"repetition_penalty": 1.05, "temperature": 0.01, "top_k": 1, "top_p": 0.85, "max_new_tokens": 512,
                   "do_sample": False, "seed": 2023}
headers = {'Content-Type': 'application/json'}
WIKI_NUM_PASSAGES = 20
BING_NUM_SNIPPETS = 20

baichuan_rag_prompt = "<C_Q> Now, based on the following reference and your knowledge, please answer the question more succinctly and professionally. The reference is delimited by triple brackets [[[]]]. The question is delimited by triple parentheses ((())). You should include as many possible answers as you can. Reference: [[[{context}]]], \nquestion: ((({question}))) <C_A>"
llama_rag_prompt = "<s>[INST] <<SYS>>\nNow, based on the following reference and your knowledge, please answer the question more succinctly and professionally. The reference is delimited by triple brackets [[[]]]. The question is delimited by triple parentheses ((())). You should include as many possible answers as you can. \n<</SYS>>\nReference: [[[{context}]]], \nquestion: ((({question}))) [/INST]"
llama_rag_prompt_fluency = "<s>[INST] <<SYS>>\nNow, based on the following reference and your knowledge, please answer the question more succinctly and professionally. The reference is delimited by triple brackets [[[]]]. The question is delimited by triple parentheses ((())). You are not allowed to add fabrications or hallucinations. \n<</SYS>>\nReference: [[[{context}]]], \nquestion: ((({question}))) [/INST]"
llama_rag_cot = "<s>[INST] <<SYS>>\nNow, based on the following reference and your knowledge, please answer the question more succinctly and professionally. The reference is delimited by triple brackets [[[]]]. The question is delimited by triple parentheses ((())). You should include as many possible answers as you can. \n<</SYS>>\nReference: [[[{context}]]], \nquestion: ((({question}))) think step by step to answer the question [/INST]"
qwen_rag_prompt = "<<SYS>>\nNow, based on the following reference and your knowledge, please answer the question more succinctly and professionally. The reference is delimited by triple brackets [[[]]]. The question is delimited by triple parentheses ((())). You should include as many possible answers as you can. \n<</SYS>>\nReference: [[[{context}]]], \nquestion: ((({question})))"
qwen_rag_prompt_fluency = "<<SYS>>\nNow, based on the following reference and your knowledge, please answer the question more succinctly and professionally. The reference is delimited by triple brackets [[[]]]. The question is delimited by triple parentheses ((())). You are not allowed to add fabrications or hallucinations. \n<</SYS>>\nReference: [[[{context}]]], \nquestion: ((({question})))"
qwen_rag_cot = "<<SYS>>\nNow, based on the following reference and your knowledge, please answer the question more succinctly and professionally. The reference is delimited by triple brackets [[[]]]. The question is delimited by triple parentheses ((())). You should include as many possible answers as you can. \n<</SYS>>\nReference: [[[{context}]]], \nquestion: ((({question}))) think step by step to answer the question"
tinyllama_rag_prompt = "<|system|>\nNow, based on the following reference and your knowledge, please answer the question more succinctly and professionally. The reference is delimited by triple brackets [[[]]]. The question is delimited by triple parentheses ((())). You should include as many possible answers as you can.</s> \n<|user|> \nReference: [[[{context}]]], \nquestion: ((({question})))</s> \n<|assistant|>"
tinyllama_rag_prompt_fluency = "<|system|>\nNow, based on the following reference and your knowledge, please answer the question more succinctly and professionally. The reference is delimited by triple brackets [[[]]]. The question is delimited by triple parentheses ((())). You are not allowed to add fabrications or hallucinations.</s> \n<|user|> \nReference: [[[{context}]]], \nquestion: ((({question})))</s> \n<|assistant|>"
tinyllama_sys_rag_prompt = "Now, based on the following reference and your knowledge, please answer the question more succinctly and professionally. The reference is delimited by triple brackets [[[]]]. The question is delimited by triple parentheses ((())). You should include as many possible answers as you can."
tinyllama_sys_rag_prompt_fluency = "Now, based on the following reference and your knowledge, please answer the question more succinctly and professionally. The reference is delimited by triple brackets [[[]]]. The question is delimited by triple parentheses ((())). You are not allowed to add fabrications or hallucinations."
tinyllama_sys_prompt = "You are a helpful assistant. Please answer the question professionally."


def sending_inference_request(question):
    if "qwen72b" in chat_model and local_model is not None:
        response, history = local_model.chat(query=question, history=None)
        return response
    elif "llama70b" in chat_model:
        data = {"inputs": question, "parameters": params_baichuan}
        patience = 5
        while True:
            try:
                request = requests.post(INFERENCE_URL, json=data, headers=headers)
                res = json.loads(request.text)[0]['generated_text']
                return res
                break
            except Exception as e:
                print(f"request llama7b failed, error: {request.text}")
                patience -= 1
                sleep(1)
                if patience <= 0:
                    break
        return ""
    else:
        data = {"inputs": question, "parameters": params_baichuan}
        request = requests.post(INFERENCE_URL, json=data, headers=headers)

        return json.loads(request.text)[0]['generated_text']


def chat_prompt_search(data_line, chat_model):
    search_results = data_line["search_results"]
    assert round == len(search_results), f"round {round} and search result length {len(search_results)} not match"
    _context = data_line[f"search_results"][round - 1][:WIKI_NUM_PASSAGES]
    context = "\n".join(_context)
    question = data_line["question"]
    if "llama" in chat_model and "tiny" not in chat_model:
        prompt = llama_rag_prompt.format(context=context, question=question)
        while len(tokenizer.encode(prompt)) > 3000:
            #  remove the last 2 sentences
            _context = _context[:-2]
            context = "\n".join(_context)
            prompt = llama_rag_prompt.format(context=context, question=question)
        return prompt
    elif "qwen" in chat_model:
        prompt = qwen_rag_prompt_fluency.format(context=context, question=question)
        if dataset in ["webglm-qa", "dolly", "eli5"]:
            #  long form qa dataset, truncate the prompt to prevent memory error
            while len(tokenizer.encode(prompt)) > 3000:
                #  remove the last 2 sentences
                _context = _context[:-2]
                context = "\n".join(_context)
                prompt = qwen_rag_prompt_fluency.format(context=context, question=question)
        return prompt
    else:
        raise NotImplementedError(f"chat model {chat_model} not implemented")


if __name__ == '__main__':
    #  init 4 threads
    #  each thread has a session
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--dataset", type=str, default="webglm-qa")
    arg_parser.add_argument("--split", type=str, default="test")
    arg_parser.add_argument("--mini_dataset", action="store_true", help="whether to use mini dataset")
    arg_parser.add_argument("--chat_model", type=str, default="llama13b")
    arg_parser.add_argument("--round", type=int, default=1)
    arg_parser.add_argument("--inference_url", type=str, default="")
    args = arg_parser.parse_args()
    dataset = args.dataset
    split = args.split
    chat_model = args.chat_model
    round = args.round
    INFERENCE_URL = args.inference_url

    input_file = f"./user_intent_data/{dataset}/{chat_model}/intergen/{chat_model}round{round - 1}-{dataset}-{split}.jsonl"
    output_file = f"./user_intent_data/{dataset}/{chat_model}/intergen/{chat_model}round{round}chat-{dataset}-{split}.jsonl"


    if "llama70b" in chat_model:
        from transformers import AutoTokenizer, pipeline

        tokenizer = AutoTokenizer.from_pretrained("../../huggingface/llama-2-13b-chat-hf/")
        print(f"loaded tokenizer {tokenizer}")

    if "qwen72b" in chat_model:
        from transformers import AutoTokenizer, AutoModelForCausalLM

        tokenizer = AutoTokenizer.from_pretrained("../../huggingface/Qwen-72B-Chat/", trust_remote_code=True)
        print(f"loaded tokenizer {tokenizer}")
        #  load model locally
        qwen_model_names = {
            "qwen7b": "Qwen-7B-Chat",
            "qwen14b": "Qwen-14B-Chat",
            "qwen72b": "Qwen-72B-Chat",
        }
        from vllm_wrapper import vLLMWrapper

        gpu_device_count = torch.cuda.device_count()
        local_model = vLLMWrapper(f'../../huggingface/{qwen_model_names[chat_model]}/',
                                  tensor_parallel_size=gpu_device_count)
        print(f"loaded model {qwen_model_names[chat_model]}")

    with open(output_file, "w", encoding="utf-8") as f:
        pass
    with open(input_file, "r", encoding="utf-8") as f:
        data_lines = [json.loads(line) for line in f]

    if args.mini_dataset:
        data_lines = data_lines[:10]

    if dataset in ["webglm-qa", "dolly", "eli5"]:
        llama_rag_prompt = llama_rag_prompt_fluency
        qwen_rag_prompt = qwen_rag_prompt_fluency
        tinyllama_rag_prompt = tinyllama_rag_prompt_fluency
        tinyllama_sys_rag_prompt = tinyllama_sys_rag_prompt_fluency
        print(f"using long answer dataset {dataset}, change to fluency prompt for rag.")

    print(f"itergen chat chat model: {chat_model}, round: {round}, dataset: {dataset}, split: {split}")
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        for idx, data_line in tqdm(enumerate(data_lines), total=len(data_lines),
                                   desc=f"dataset: {dataset}, split: {split}"):
            if f"{chat_model}_round_answer" not in data_lines[idx]:
                data_lines[idx][f"{chat_model}_round_answer"] = []
            if data_line["search_results"][round - 1] == "":
                data_lines[idx][f"{chat_model}_round_answer"].append("")
            else:
                model_input = chat_prompt_search(data_line, chat_model)
                if idx == 1:
                    print(model_input)
                future_res = executor.submit(sending_inference_request, model_input)
                res = future_res.result()
                data_lines[idx][f"{chat_model}_round_answer"].append(res)

    with open(output_file, "a", encoding="utf-8") as f:
        #  write to file
        for data_line in data_lines:
            f.write(json.dumps(data_line, ensure_ascii=False) + "\n")
