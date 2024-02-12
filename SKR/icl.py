import argparse
import re

import requests
import json
import os
import concurrent.futures

import torch
from tqdm import tqdm

params_baichuan = {"repetition_penalty": 1.05, "temperature": 0.01, "top_k": 1, "top_p": 0.85, "max_new_tokens": 512,
                   "do_sample": False, "seed": 2023}
headers = {'Content-Type': 'application/json'}

llama_known_prompt = """<s>[INST] <<SYS>>
Based on your knowledge you should tell if you need additional information to answer the following questions.
<</SYS>>
Question: [[What C is the alternative name for the Water Moccasin,the U.S.A.’s only venomous water snake?]] Do you need additional information to answer this question? [/INST] No, I don't need additional information. </s><s> [INST] 
Question: [[what age do you have to be to smoke in france?]] Do you need additional information to answer this question? [/INST] Yes, I need additional information. </s><s> [INST]
Question: [[who was the slytherin seeker in harry potter 1?]] Do you need additional information to answer this question? [/INST] No, I don't need additional information. </s><s> [INST]
Question: [[{question}]] Do you need additional information to answer this question? [/INST]"""

qwen_known_prompt = """Based on your knowledge you should tell if you need additional information to answer the following questions.
Question: [[What C is the alternative name for the Water Moccasin,the U.S.A.’s only venomous water snake?]] Do you need additional information to answer this question? No, I don't need additional information. 
Question: [[what age do you have to be to smoke in france?]] Do you need additional information to answer this question? Yes, I need additional information. 
Question: [[who was the slytherin seeker in harry potter 1?]] Do you need additional information to answer this question? No, I don't need additional information. 
Question: [[{question}]] Do you need additional information to answer this question? """

def sending_inference_request(question):
    if "qwen72b" in answer_model and local_model is not None:
        response, history = local_model.chat(query=question, history=None)
        return response
    else:
        data = {"inputs": question, "parameters": params_baichuan}
        request = requests.post(INFERENCE_URL, json=data, headers=headers)

        return json.loads(request.text)[0]['generated_text']


if __name__ == '__main__':
    #  init 4 threads
    #  each thread has a session
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--dataset", type=str, default="webglm-qa")
    arg_parser.add_argument("--split", type=str, default="test")
    arg_parser.add_argument("--mini_dataset", action="store_true", help="whether to use mini dataset")
    arg_parser.add_argument("--answer_model", type=str, default="llama13b")
    arg_parser.add_argument("--rewrite_model", type=str, default="llama13b")
    arg_parser.add_argument("--inference_url", type=str, default="")
    args = arg_parser.parse_args()
    dataset = args.dataset
    split = args.split
    answer_model = args.answer_model
    rewrite_model = args.rewrite_model
    INFERENCE_URL = args.inference_url
    prompt_method = "without_search"
    custom_rewrite_methods = ["llama_rewrite_search", "rule_rewrite_search"]

    print(f"INFERENCE_URL: {INFERENCE_URL}")

    if not os.path.exists(f"./user_intent_data/{dataset}/rewrite/{rewrite_model}/"):
        os.makedirs(f"./user_intent_data/{dataset}/rewrite/{rewrite_model}/", exist_ok=True)
    input_file = f"./user_intent_data/{dataset}/{dataset}-{split}.jsonl"
    output_file = f"./user_intent_data/{dataset}/rewrite/{rewrite_model}/unparsed-{rewrite_model}-{dataset}-{split}.jsonl"

    with open(output_file, "w", encoding="utf-8") as f:
        pass
    with open(input_file, "r", encoding="utf-8") as f:
        data_lines = [json.loads(line) for line in f]

    #  tokenize input for llama to truncate inputs which are too long
    tokenizer = None
    if "llama" in answer_model:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("../../huggingface/llama-2-13b-chat-hf/")
        print(f"loaded tokenizer {tokenizer}")
    elif "qwen72b" in answer_model:
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
        local_model = vLLMWrapper(f'../../huggingface/{qwen_model_names[answer_model]}/',
                                  tensor_parallel_size=gpu_device_count)
        print(f"loaded model {qwen_model_names[answer_model]}")

    if args.mini_dataset:
        data_lines = data_lines[:10]
    #  trim dataset that is too large
    data_lines = data_lines[:5000]
    print(f"dataset: {dataset}, split: {split}, rewrite_model: {rewrite_model}")
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        for idx, data_line in tqdm(enumerate(data_lines), total=len(data_lines),
                                   desc=f"dataset: {dataset}, split: {split}"):
            #  do search with vanilla query
            if "baichuan" in answer_model:
                model_input = "<C_Q> " + data_line["question"] + " <C_A>"
            elif "llama" in answer_model:
                model_input = llama_known_prompt.format(question=data_line["question"])
            elif "qwen72b" in answer_model:
                model_input = qwen_known_prompt.format(question=data_line["question"])
            else:
                raise NotImplementedError(f"chat model {answer_model} not implemented")

            if idx == 1:
                print(model_input)
            future_res = executor.submit(sending_inference_request, model_input)
            res = future_res.result()
            data_lines[idx][f"known"] = res

    with open(output_file, "a", encoding="utf-8") as f:
        #  write to file
        for data_line in data_lines:
            f.write(json.dumps(data_line, ensure_ascii=False) + "\n")

