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
    if "qwen7b" in chat_model and local_model is not None:
        response, history = local_model.chat(tokenizer, question, history=None)
        return response
    if "tinyllama" in chat_model and local_model is not None:
        messages = [
            {
                "role": "system",
                "content": tinyllama_rag_prompt,
            },
            {"role": "user", "content": question},
        ]
        # prompt = local_model.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # outputs = local_model(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
        # res = outputs[0]["generated_text"]
        # input_len = res.find("<|assistant|>")
        # return res[input_len + len("<|assistant|>"):].strip()
        return None
    if "phi2" in chat_model:
        inputs = tokenizer(question, return_tensors="pt", return_attention_mask=False)
        input_len = len(inputs["input_ids"][0])
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        outputs = model.generate(**inputs, do_sample=False, max_new_tokens=512)
        text = tokenizer.batch_decode(outputs[:, input_len:], skip_special_tokens=True)[0]
        return text
    elif "llama70b" in chat_model:
        data = {"inputs": question, "parameters": params_baichuan}
        patience=5
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


def chat_prompt_vanilla_search(data_line, search_engine, chat_model):
    if search_engine == "bing":
        _context = [item["snippet"] for item in data_line["vanilla_search_results"][:BING_NUM_SNIPPETS]]
    elif search_engine == "kiltbm25":
        _context = data_line["vanilla_search_results"][:WIKI_NUM_PASSAGES]
    else:
        raise NotImplementedError(f"search engine {search_engine} not implemented")
    context = "\n".join(_context)
    question = data_line["question"]
    if "baichuan" in chat_model:
        prompt = baichuan_rag_prompt.format(context=context, question=question)
    if "tinyllama" in chat_model:
        prompt = tinyllama_rag_prompt.format(context=context, question=question)
    elif "llama" in chat_model and "tiny" not in chat_model:
        prompt = llama_rag_prompt.format(context=context, question=question)
        while len(tokenizer.encode(prompt)) > 3000:
            #  remove the last 2 sentences
            _context = _context[:-2]
            context = "\n".join(_context)
            prompt = llama_rag_prompt.format(context=context, question=question)
    elif "qwen" in chat_model:
        prompt = qwen_rag_prompt.format(context=context, question=question)
        if dataset in ["webglm-qa", "dolly", "eli5"]:
            #  long form qa dataset, truncate the prompt to prevent memory error
            while len(tokenizer.encode(prompt)) > 3000:
                #  remove the last 2 sentences
                _context = _context[:-2]
                context = "\n".join(_context)
                prompt = llama_rag_prompt.format(context=context, question=question)
    else:
        raise NotImplementedError(f"chat model {chat_model} not implemented")
    return prompt


def chat_prompt_search(data_line, rewrite_model, search_engine, chat_model, use_webcontent=False):
    #  put top 1 search results into context, then top 2, then top 3, then top 4, then top 5
    if search_engine == "bing":
        #  switch column and row
        context = [[item for item in searchword_results[:BING_NUM_SNIPPETS]] for searchword_results in
                   data_line[f"{rewrite_model}_rewrite_search_results"]]
        context = list(map(list, zip(*context)))
        #  flatten list
        context = [item for sublist in context for item in sublist]
        _context = []
        _context_url = set()
        #  get references with different ids
        for i in range(len(context)):
            if len(_context_url) == BING_NUM_SNIPPETS:
                break
            if context[i]["url"] not in _context_url:
                if use_webcontent:
                    _context.append(context[i]["webcontent"])
                else:
                    _context.append(context[i]["snippet"])
                _context_url.add(context[i]["url"])
    elif search_engine == "kiltbm25":
        # take top WIKI_NUM_PASSAGES passages from each search result
        _context = data_line[f"{rewrite_model}_rewrite_search_results"][:WIKI_NUM_PASSAGES]
    else:
        raise NotImplementedError(f"search engine {search_engine} not implemented")

    if len(_context) > 0:
        context = "\n".join(_context)
        question = data_line["question"]
        if "baichuan" in chat_model:
            prompt = baichuan_rag_prompt.format(context=context, question=question)
        elif "tinyllama" in chat_model:
            prompt = tinyllama_rag_prompt.format(context=context, question=question)
        elif "llama" in chat_model and "tiny" not in chat_model:
            prompt = llama_rag_prompt.format(context=context, question=question)
            while len(tokenizer.encode(prompt)) > 3000:
                #  remove the last 2 sentences
                _context = _context[:-2]
                context = "\n".join(_context)
                prompt = llama_rag_prompt.format(context=context, question=question)
        elif "qwen" in chat_model:
            prompt = qwen_rag_prompt.format(context=context, question=question)
            if dataset in ["webglm-qa", "dolly", "eli5"]:
                #  long form qa dataset, truncate the prompt to prevent memory error
                while len(tokenizer.encode(prompt)) > 3000:
                    #  remove the last 2 sentences
                    _context = _context[:-2]
                    context = "\n".join(_context)
                    prompt = qwen_rag_prompt.format(context=context, question=question)
        else:
            raise NotImplementedError(f"chat model {chat_model} not implemented")
    else:
        question = data_line["question"]
        if "baichuan" in chat_model:
            prompt = f"<C_Q> {question} <C_A>"
        elif "tinyllama" in chat_model:
            prompt = f"<|system|>\nYou are a helpful assistant. Please answer the question professionally.</s>\n<|user|>\n{question}</s>\n<|assistant|>"
        elif "llama" in chat_model and "tiny" not in chat_model:
            prompt = f"<s>[INST] {question} [/INST]"
        elif "qwen" in chat_model:
            prompt = question
        else:
            raise NotImplementedError(f"chat model {chat_model} not implemented")
    return prompt


def gold_search(data_line):
    context = "\n".join(data_line['references'])
    question = data_line["question"]
    prompt = f"<C_Q>Now, entirely based on the following reference, please answer the question more succinctly and professionally. The reference is delimited by triple brackets [[[]]]. The question is delimited by triple parentheses ((())). You should include as many possible answers as you can. Reference: [[[{context}]]], \nquestion: ((({question})))<C_A>"

    return prompt


if __name__ == '__main__':
    #  init 4 threads
    #  each thread has a session
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--dataset", type=str, default="webglm-qa")
    arg_parser.add_argument("--split", type=str, default="test")
    arg_parser.add_argument("--prompt_method", type=str, default="vanilla_search")
    arg_parser.add_argument("--mini_dataset", action="store_true", help="whether to use mini dataset")
    arg_parser.add_argument("--chat_model", type=str, default="llama13b")
    arg_parser.add_argument("--search_engine", type=str, default="kiltbm25")
    arg_parser.add_argument("--rerank_model", type=str, default="e5base")
    arg_parser.add_argument("--inference_url", type=str, default="")
    args = arg_parser.parse_args()
    dataset = args.dataset
    split = args.split
    prompt_method = args.prompt_method
    chat_model = args.chat_model
    search_engine = args.search_engine
    rerank_model = args.rerank_model
    INFERENCE_URL = args.inference_url
    _prompt_method = prompt_method if not prompt_method.endswith("_webcontent") else prompt_method[:-11]

    custom_rewrite_methods = ["llama_rewrite_search", "rule_rewrite_search"]

    print(f"INFERENCE_URL: {INFERENCE_URL}")

    if "without_search" in prompt_method:
        if not os.path.exists(f"./user_intent_data/{dataset}/{chat_model}/{prompt_method}/"):
            os.makedirs(f"./user_intent_data/{dataset}/{chat_model}/{prompt_method}/", exist_ok=True)
        input_file = f"./user_intent_data/{dataset}/{dataset}-{split}.jsonl"
        output_file = f"./user_intent_data/{dataset}/{chat_model}/{prompt_method}/{chat_model}-{dataset}-{split}.jsonl"
    else:
        if not os.path.exists(f"./user_intent_data/{dataset}/{chat_model}/{search_engine}/{prompt_method}/"):
            os.makedirs(f"./user_intent_data/{dataset}/{chat_model}/{search_engine}/{prompt_method}/", exist_ok=True)
        if "bing" in search_engine:
            input_file = f"./user_intent_data/{dataset}/{search_engine}/{_prompt_method}/{search_engine}-{dataset}-{split}.jsonl"
            output_file = f"./user_intent_data/{dataset}/{chat_model}/{search_engine}/{prompt_method}/{chat_model}-{dataset}-{split}.jsonl"
        else:
            input_file = f"./user_intent_data/{dataset}/{search_engine}/{_prompt_method}/{rerank_model}-{search_engine}-{dataset}-{split}.jsonl"
            output_file = f"./user_intent_data/{dataset}/{chat_model}/{search_engine}/{prompt_method}/{rerank_model}-{chat_model}-{dataset}-{split}.jsonl"

    with open(output_file, "w", encoding="utf-8") as f:
        pass
    with open(input_file, "r", encoding="utf-8") as f:
        data_lines = [json.loads(line) for line in f]

    #  tokenize input for llama to truncate inputs which are too long
    tokenizer = None
    local_model = None
    # if "tinyllama" in chat_model:
    #     pipe = pipeline("text-generation", model="../../huggingface/TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.bfloat16,
    #                     device_map="auto")
    if "llama" in chat_model and "tiny" not in chat_model:
        from transformers import AutoTokenizer, pipeline

        tokenizer = AutoTokenizer.from_pretrained("../../huggingface/llama-2-13b-chat-hf/")
        print(f"loaded tokenizer {tokenizer}")
    elif "qwen72b" in chat_model:
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
    elif "qwen7b" in chat_model:
        from transformers import AutoTokenizer, AutoModelForCausalLM

        tokenizer = AutoTokenizer.from_pretrained("../../huggingface/Qwen-7B-Chat/", trust_remote_code=True)
        local_model = AutoModelForCausalLM.from_pretrained(f"../../huggingface/Qwen-7B-Chat/", device_map="auto", trust_remote_code=True)
        if torch.cuda.is_available():
            local_model = local_model.cuda()
        local_model.eval()
    elif "phi2" in chat_model:
        if torch.cuda.is_available():
            torch.set_default_device("cuda")
        model = AutoModelForCausalLM.from_pretrained("../../huggingface/phi2", torch_dtype="auto",
                                                     trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained("../../huggingface/phi2", trust_remote_code=True)
        model.eval()
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
        print(f"loaded tokenizer {tokenizer}")
        print(f"loaded model {model}")

    if args.mini_dataset:
        data_lines = data_lines[:20]
    #  trim dataset that is too large
    data_lines = data_lines[:5000]
    print(
        f"dataset: {dataset}, split: {split}, prompt_method: {prompt_method}, chat_model: {chat_model}, search_engine: {search_engine}, rerank_model: {rerank_model}")
    if dataset in ["webglm-qa", "dolly", "eli5"]:
        llama_rag_prompt = llama_rag_prompt_fluency
        qwen_rag_prompt = qwen_rag_prompt_fluency
        tinyllama_rag_prompt = tinyllama_rag_prompt_fluency
        tinyllama_sys_rag_prompt = tinyllama_sys_rag_prompt_fluency
        print(f"using long answer dataset {dataset}, change to fluency prompt for rag.")
    if dataset in ["2wiki", "musique"] and "qwen" in chat_model:
        qwen_rag_prompt = qwen_rag_cot
        print(f"using multi-hop answer dataset {dataset}, change to cot prompt for rag.")
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        for idx, data_line in tqdm(enumerate(data_lines), total=len(data_lines),
                                   desc=f"dataset: {dataset}, split: {split}"):
            #  do search with vanilla query
            if "vanilla_search" in prompt_method:
                model_input = chat_prompt_vanilla_search(data_line, search_engine, chat_model)
            elif "gpt4_rewrite_search" in prompt_method:
                use_webcontent = "webcontent" in prompt_method
                model_input = chat_prompt_search(data_line, "gpt4", search_engine, chat_model,
                                                 use_webcontent=use_webcontent)

            elif prompt_method == "without_search":
                #  do not use rag prompt for without search
                tinyllama_sys_rag_prompt = tinyllama_sys_prompt
                if "baichuan" in chat_model:
                    model_input = "<C_Q> " + data_line["question"] + " <C_A>"
                elif "tinyllama" in chat_model:
                    model_input = f"<|system|>\nYou are a helpful assistant. Please answer the question professionally and provide explanations.</s>\n<|user|>\n{data_line['question']}</s>\n<|assistant|>"
                elif "llama" in chat_model and "tiny" not in chat_model:
                    model_input = "<s>[INST] " + data_line["question"] + " [/INST]"
                elif "qwen7b" in chat_model:
                    model_input = f"Please answer the question professionally and provide explanations. Question(({data_line['question']}))"
                elif "qwen72b" in chat_model:
                    model_input = data_line["question"]
                elif "phi2" in chat_model:
                    model_input = f"Instruct: Please answer the question professionally and provide explanations. question(({data_line['question']}))\nOutput:"
                else:
                    raise NotImplementedError(f"chat model {chat_model} not implemented")
            elif re.match(r"v\d[0-9a-z]+_rewrite_search",
                          prompt_method) or prompt_method in custom_rewrite_methods or "plus" in prompt_method:
                model_input = chat_prompt_search(data_line, prompt_method.replace("_rewrite_search", ""), search_engine,
                                                 chat_model)
            elif prompt_method == "gold_search":
                model_input = gold_search(data_line)

            else:
                raise NotImplementedError(f"prompt method {prompt_method} not implemented")
            if idx == 1:
                print(model_input)
            future_res = executor.submit(sending_inference_request, model_input)
            res = future_res.result()
            data_lines[idx][f"{chat_model}_{prompt_method}_answer"] = res

    with open(output_file, "a", encoding="utf-8") as f:
        #  write to file
        for data_line in data_lines:
            f.write(json.dumps(data_line, ensure_ascii=False) + "\n")
