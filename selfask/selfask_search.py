import argparse
import json
import os
from time import sleep

import requests
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM, LlamaTokenizer, AutoModel
import torch.nn.functional as F
from torch import Tensor

params_query_rewrite = {"repetition_penalty": 1.05, "temperature": 0.01, "top_k": 1, "top_p": 0.85,
                        "max_new_tokens": 512,
                        "do_sample": False, "seed": 2023}
headers = {'Content-Type': 'application/json'}
intermediate = "Intermediate answer:"
followup = "Follow up:"
finalans = 'So the final answer is:'


def parse_chat_result(output_str):
    if intermediate in output_str and followup in output_str:
        query = output_str.split(intermediate)[0].split(followup)[1:]
        query = "".join(query)
        return {"query": query, "answer": ""}
    elif finalans in output_str:
        answer = output_str.split(finalans)[1]
        return {"query": "", "answer": answer}
    else:
        # print(f"output_str: {output_str}")
        return {"query": "", "answer": ""}


def apply_retrieval(queries, question):
    #  the question is a string
    if search_engine == "kiltbm25":
        patience = 3
        search_results = []
        while True:
            try:
                body = {'queries': queries, 'question': question}
                if search_engine == "kiltbm25":
                    res = requests.request('GET', search_api, json=body)
                else:
                    raise NotImplementedError(f"search engine {search_engine} is not implemented")
                search_results = json.loads(res.text)["search_results"]
                break
            except Exception as e:
                print(f"search query {question} failed, error: {e}")
                patience -= 1
                sleep(1)
                if patience <= 0:
                    break
        return search_results
    else:
        raise NotImplementedError(f"search engine {search_engine} is not implemented")


def retrieval_round(data_lines):
    # calculate the logits
    for idx, chat in enumerate(tqdm(data_lines)):
        if "final_answer" in chat:
            chat["query"].append("")
            chat["search_results"].append([])
            continue
        question = chat["question"]
        chat_result = data_lines[idx][f"{chat_model}_round_answer"][round]
        #  check the chat result
        chat_result = parse_chat_result(chat_result)
        query = chat_result["query"]
        final_answer = chat_result["answer"]
        # conduct retrieval
        #  if query is empty, skip retrieval
        if "search_results" not in chat:
            chat["query"] = []
            chat["search_results"] = []
        if not query:
            chat["query"].append("")
            chat["search_results"].append([])
            chat_result["final_answer"] = final_answer
        else:
            #  do retrieval
            search_results = apply_retrieval([query], question)
            chat["query"].append(query)
            chat["search_results"].append(search_results)
    return data_lines


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--dataset", type=str, default="webglm-qa")
    arg_parser.add_argument("--split", type=str, default="test")
    arg_parser.add_argument("--chat_model", type=str, default="llama13b")
    arg_parser.add_argument("--rewrite_model", type=str, default="v0108")
    arg_parser.add_argument("--search_engine", type=str, default="kiltbm25")
    arg_parser.add_argument("--local_inference", action="store_true")
    arg_parser.add_argument("--address", type=str, default="")
    arg_parser.add_argument("--round", type=int, default=0)
    arg_parser.add_argument("--mini_dataset", action="store_true")
    arg_parser.add_argument("--search_api", type=str, default="http://172.17.31.12:5050/search/bm25_e5_rerank/")
    args = arg_parser.parse_args()
    dataset = args.dataset
    split = args.split
    rewrite_model = args.rewrite_model
    chat_model = args.chat_model
    search_engine = args.search_engine
    round = args.round
    address = args.address
    search_api = args.search_api

    rerank_model = "e5base"

    local_model = None
    if args.local_inference:
        model_path = [file for file in os.listdir("./models") if rewrite_model in file][0]
        model_path = os.path.join("./models", model_path)
        print(f"local model path: {model_path}")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        local_model = AutoModelForCausalLM.from_pretrained(model_path).eval()
        if torch.cuda.is_available():
            local_model.cuda()
        local_tokenizer = AutoTokenizer.from_pretrained(model_path)

    print(
        f"data: {dataset}, split: {split}, chat model: {chat_model}, rewrite model: {rewrite_model}, search engine: {search_engine}, round: {round}")

    output_file = f"./user_intent_data/{dataset}/{chat_model}/selfask/{chat_model}round{round}-{dataset}-{split}.jsonl"
    if not os.path.exists(f"./user_intent_data/{dataset}/{chat_model}/selfask/"):
        os.makedirs(f"./user_intent_data/{dataset}/{chat_model}/selfask/", exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        pass

    previous_round_chat_file = f"./user_intent_data/{dataset}/{chat_model}/selfask/{chat_model}round{round}chat-{dataset}-{split}.jsonl"
    data_lines = [json.loads(line) for line in open(previous_round_chat_file, "r", encoding="utf-8")]
    if args.mini_dataset:
        data_lines = data_lines[:10]

    data_lines = retrieval_round(data_lines)
    #  save the result
    with open(output_file, "w", encoding="utf-8") as f:
        for line in data_lines:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
