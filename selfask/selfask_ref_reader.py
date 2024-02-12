import argparse
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

sys_prompt = """Given the following reference, answer it by a brief sentence.  You are not allowed to add fabrications or hallucinations."""

demo = '''Question:  How old was Muhammad Ali when he died?
Answer: Muhammad Ali was 74 years old when he died.

Question: Who was the founder of craigslist?
Answer: Craigslist was founded by Craig Newmark.

Question: Who was the father of Mary Ball Washington?
Answer: The father of Mary Ball Washington was Joseph Ball.

Question: Who is the director of Casino Royale? 
Answer: The director of Casino Royale is Martin Campbell. 

Question: '''

llama_rag_prompt = "<s>[INST] <<SYS>>\n{sys_prompt} \n<<SYS>>{demo}\n{question}\nAnswer:[/INST]"
qwen_rag_prompt = "<<SYS>>\n{sys_prompt} \n<<SYS>>\n{demo}\n{question}\nAnswer:"


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
    question = data_line["question"]
    if round == 0:
        history = ""
    else:
        history = data_line[f"history"][-1]
    if "llama" in chat_model and "tiny" not in chat_model:
        prompt = llama_rag_prompt.format(demo=demo, question=question, history=history)
    elif "qwen" in chat_model:
        prompt = qwen_rag_prompt.format(demo=demo, question=question, history=history)
    else:
        raise NotImplementedError(f"chat model {chat_model} not implemented")
    return prompt


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

    if round == 0:
        input_file = f"./user_intent_data/{dataset}/{chat_model}/without_search/{chat_model}-{dataset}-{split}.jsonl"
    else:
        input_file = f"./user_intent_data/{dataset}/{chat_model}/selfask/{chat_model}round{round - 1}-{dataset}-{split}.jsonl"
    output_file = f"./user_intent_data/{dataset}/{chat_model}/selfask/{chat_model}round{round}chat-{dataset}-{split}.jsonl"
    if not os.path.exists(f"./user_intent_data/{dataset}/{chat_model}/selfask/"):
        os.makedirs(f"./user_intent_data/{dataset}/{chat_model}/selfask/", exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        pass
    with open(input_file, "r", encoding="utf-8") as f:
        data_lines = [json.loads(line) for line in f]

    if args.mini_dataset:
        data_lines = data_lines[:10]

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

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        for idx, data_line in tqdm(enumerate(data_lines), total=len(data_lines),
                                   desc=f"dataset: {dataset}, split: {split}"):
            model_input = chat_prompt_search(data_line, chat_model)
            if idx == 0:
                print(f"model input: {model_input}")
            future_res = executor.submit(sending_inference_request, model_input)
            res = future_res.result()
            if f"{chat_model}_round_answer" not in data_lines[idx]:
                data_lines[idx][f"{chat_model}_round_answer"] = []
            data_lines[idx][f"{chat_model}_round_answer"].append(res)

    with open(output_file, "a", encoding="utf-8") as f:
        #  write to file
        for data_line in data_lines:
            f.write(json.dumps(data_line, ensure_ascii=False) + "\n")
