import argparse
import concurrent.futures
import json

import requests
import torch
from tqdm import tqdm
from time import sleep

params_baichuan = {"repetition_penalty": 1.05, "temperature": 0.01, "top_k": 1, "top_p": 0.85, "max_new_tokens": 512,
                   "do_sample": False, "seed": 2023}
headers = {'Content-Type': 'application/json'}

llama_flare_prompt = "<s>[INST] Search results: \n{search_results} \n{question} \n[/INST] \n{history} "
qwen_flare_prompt = "Search results: \n{search_results} \n{question}"



def sending_inference_request(question, search_results, history, display=False):
    #  the input search results is a list of strings, each string is a search result
    #  search results is in the format like "[1] result1\n[2] result2\n[3] result3..."
    #  append [x] to each search result
    if "qwen72b" in chat_model:
        flare_prompt = qwen_flare_prompt
    elif "llama" in chat_model:
        flare_prompt = llama_flare_prompt
    else:
        raise NotImplementedError(f"chat_model {chat_model} not implemented")
    _search_results = "\n".join([f"[{idx + 1}] {result}" for idx, result in enumerate(search_results)])
    model_input = flare_prompt.format(search_results=_search_results, question=question, history=history)
    while len(tokenizer.encode(model_input)) > 3000:
        #  remove the last 2 sentences
        search_results = search_results[:-2]
        # print(f"search results too long, remove the last 2 sentences")
        _search_results = "\n".join([f"[{idx + 1}] {result}" for idx, result in enumerate(search_results)])
        model_input = flare_prompt.format(search_results=_search_results, question=question, history=history)

    if "qwen72b" in chat_model and local_model is not None:
        if display:
            print(f"model input: {model_input}")
            print(f"history: {history}")
        response, _ = local_model.chat(query=model_input, history=history, chat_format="continue-chatml")
        return response
    else:
        if display:
            print(f"model input: {model_input}")
        data = {"inputs": model_input, "parameters": params_baichuan}
        patience = 5
        while True:
            try:
                request = requests.post(INFERENCE_URL, json=data, headers=headers)
                res = json.loads(request.text)['generated_text']
                return res
                break
            except Exception as e:
                print(f"request llama7b failed, error: {request.text}")
                patience -= 1
                sleep(1)
                if patience <= 0:
                    break
        return ""

def first_round_inference_req(question):
    if "qwen72b" in chat_model and local_model is not None:
        model_input = question
        response, logits = local_model.chat(query=model_input, chat_format="continue-chatml")
        return response, logits
    else:
        model_input=f"<s>[INST] {question} [/INST]"
        data = {"inputs": model_input, "parameters": params_baichuan}
        request = requests.post(INFERENCE_URL, json=data, headers=headers)
        return json.loads(request.text)[0]['generated_text'], None


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--dataset", type=str, default="webglm-qa")
    arg_parser.add_argument("--split", type=str, default="test")
    arg_parser.add_argument("--chat_model", type=str, default="llama13b")
    arg_parser.add_argument("--inference_url", type=str, default="http://localhost:8000")
    arg_parser.add_argument("--mini_dataset", action="store_true")
    arg_parser.add_argument("--round", type=int, default=1)

    args = arg_parser.parse_args()
    dataset = args.dataset
    split = args.split
    chat_model = args.chat_model
    INFERENCE_URL = args.inference_url
    round = args.round

    print(f"INFERENCE_URL: {INFERENCE_URL}")

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
        local_model = vLLMWrapper(f'../../huggingface/{qwen_model_names[chat_model]}/', tensor_parallel_size=gpu_device_count)
        print(f"loaded model {qwen_model_names[chat_model]}")
    else:
        raise NotImplementedError(f"chat_model {chat_model} not implemented")

    if round == 0:
        raise ValueError("round 0 is not supported")
        # input_file = f"./user_intent_data/{dataset}/{dataset}-{split}.jsonl"
        # output_file = f"./user_intent_data/{dataset}/{chat_model}/flare/{chat_model}round{round}chat-{dataset}-{split}.jsonl"
        #
        # with open(output_file, "w", encoding="utf-8") as f:
        #     pass
        # datalines = [json.loads(line) for line in open(input_file, "r", encoding="utf-8")]
        # if args.mini_dataset:
        #     datalines = datalines[:10]
        # with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        #     for idx, data_line in tqdm(enumerate(datalines), total=len(datalines),
        #                                desc=f"dataset: {dataset}, split: {split}"):
        #         question = data_line["question"]
        #         future_res = executor.submit(first_round_inference_req, question)
        #         if idx == 0:
        #             print(f"question: {question}")
        #         data_line[f"{chat_model}_round{round}_answer"], data_line[f"{chat_model}_round{round}_logits"] = future_res.result()
    if round >= 1:
        input_file = f"./user_intent_data/{dataset}/{chat_model}/flare/{chat_model}round{round - 1}-{dataset}-{split}.jsonl"
        output_file = f"./user_intent_data/{dataset}/{chat_model}/flare/{chat_model}round{round}chat-{dataset}-{split}.jsonl"

        last_round_chat_result = [json.loads(line) for line in open(input_file, "r", encoding="utf-8")]
        if args.mini_dataset:
            last_round_chat_result = last_round_chat_result[:10]
        print(f"flare dataset: {dataset}, split: {split}, chat_model: {chat_model}, round: {round}")
        with open(output_file, "w", encoding="utf-8") as f:
            pass

        display = True
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            for idx, data_line in tqdm(enumerate(last_round_chat_result), total=len(last_round_chat_result),
                                       desc=f"dataset: {dataset}, split: {split}"):
                #  check if the last round query is empty
                former_round_queries = data_line["query"]
                if len(former_round_queries) < round:
                    raise ValueError(f"former_round_queries {former_round_queries} is not enough for round {round}")

                former_round_query = former_round_queries[round - 1]
                if f"{chat_model}_round_answer" not in data_line:
                    data_line[f"{chat_model}_round_answer"] = []
                if former_round_query == "":
                    # stop for this round
                    data_line[f"{chat_model}_round_answer"].append("")
                else:
                    history = "".join(data_line["history"])
                    search_results = data_line["search_results"][round - 1]
                    question = data_line["question"]
                    if display:
                        future_res = executor.submit(sending_inference_request, question, search_results,
                                                     history, display)
                        display = False
                    else:
                        future_res = executor.submit(sending_inference_request, question, search_results,
                                                     history, display)
                    data_line[f"{chat_model}_round_answer"].append(future_res.result())

    with open(output_file, "a", encoding="utf-8") as f:
        for data_line in last_round_chat_result:
            f.write(json.dumps(data_line) + "\n")
