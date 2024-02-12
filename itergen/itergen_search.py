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


def parse_query_rewrite_result(question, output_str):
    #  parse task
    try:
        task = output_str.split("<Task(")[1].split(")>")[0]
    except:
        print(f"parse timeliness error: {output_str}")
        task = ""
    #  parse timeliness
    try:
        timeliness = output_str.split("<Timeliness(")[1].split(")>")[0]
        timeliness = True if timeliness == "True" else False
    except:
        print(f"parse timeliness error: {output_str}")
        timeliness = False

    #  parse questions
    try:
        questions = output_str.split("<Questions>")[1].split("</Questions>")[0]
        questions = [q for q in questions.split("<Question(") if q.strip()]
        rewrite_questions = []

        for question_block in questions:
            question = question_block.split(")>")[0]
            need_search = question_block.split("<NeedSearch(")[1].split(")>")[0]
            need_search = True if need_search == "True" else False
            search_word = question_block.split("<Query(")[1].split(")>")[0]
            rewrite_questions.append({
                "question": question,
                "needSearch": need_search,
                "query": search_word
            })
    except:
        print(f"parse questions error: {output_str}")
        rewrite_questions = []

    #  parse claims
    try:
        claims = output_str.split("<Claims>")[1].strip()
        if claims.endswith("</Claims>"):
            claims = claims.split("</Claims>")[0]
        claims = [c for c in claims.split("<Claim(") if c.strip()]
        rewrite_claims = []
        for claim_block in claims:
            claim_block = claim_block.strip()
            if not claim_block.endswith(")>") or "<NeedSearch(" not in claim_block or "<Query(" not in claim_block:
                continue
            claim = claim_block.split(")>")[0]
            need_search = claim_block.split("<NeedSearch(")[1].split(")>")[0]
            need_search = True if need_search == "True" else False
            query = claim_block.split("<Query(")[1].split(")>")[0]
            rewrite_claims.append({
                "claim": claim,
                "needSearch": need_search,
                "query": query
            })
    except:
        print(f"parse claims error: {output_str}")
        rewrite_claims = []

    parse_res = {
        "task": task,
        "timeliness": timeliness,
        "questions": rewrite_questions,
        "claims": rewrite_claims
    }
    queries = [question] + [q["query"] for q in parse_res["questions"] if q["needSearch"]]
    queries += [q["query"] for q in parse_res["claims"] if q["needSearch"] and "query" in q and q["query"] != ""]
    return queries


def sending_rewrite_request(address, question, answer):
    prompt = (f"<s>[INST] <<SYS>>\nYou are a helpful assistant. Your task is to parse user input into"
              f" structured formats according to the coarse answer. Current datatime is 2023-12-20 9:47:28"
              f" <</SYS>>\n Course answer: (({answer}))\nQuestion: "
              f"(({question})) [/INST]")

    if local_model is not None and args.local_inference:
        input_ids = local_tokenizer.encode(prompt, return_tensors="pt")
        len_input_ids = len(input_ids[0])
        if torch.cuda.is_available():
            input_ids = input_ids.cuda()
        outputs = local_model.generate(input_ids)
        res = local_tokenizer.decode(outputs[0][len_input_ids:], skip_special_tokens=True)
        return parse_query_rewrite_result(question, res)
    else:
        data = {"inputs": prompt, "parameters": params_query_rewrite}
        request = requests.post(f"http://{address}", json=data, headers=headers)
        return parse_query_rewrite_result(question, json.loads(request.text)[0]['generated_text'])


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
        question = chat["question"]
        if round == 0:
            answer = chat[f"{chat_model}_vanilla_search_answer"]

        else:
            answer = chat[f"{chat_model}_round_answer"][round - 1]
            # #  aggregate the all former round queries
            # former_queries = [q for q in chat["query"] if q]
            # # flatten the list
            # former_queries = [item for sublist in former_queries for item in sublist]
        #  check the chat result
        # queries = sending_rewrite_request(address, question, answer)
        # # print(f"queries: {queries}")
        # # print(f"former queries: {former_queries}")
        # queries = list(set(queries + former_queries))
        query = " ".join([question, answer])
        # conduct retrieval
        #  if query is empty, skip retrieval
        if idx == 0:
            print(f"query: {query}")
        if "search_results" not in chat:
            chat["query"] = []
            chat["search_results"] = []
        if len(query) == 0:
            chat["query"].append([])
            chat["search_results"].append("")
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
        f"itergen search data: {dataset}, split: {split}, chat model: {chat_model}, rewrite model: {rewrite_model}, search engine: {search_engine}, round: {round}")

    output_file = f"./user_intent_data/{dataset}/{chat_model}/intergen/{chat_model}round{round}-{dataset}-{split}.jsonl"
    if not os.path.exists(f"./user_intent_data/{dataset}/{chat_model}/intergen/"):
        os.makedirs(f"./user_intent_data/{dataset}/{chat_model}/intergen/", exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        pass
    if round == 0:
        #  open without retrieval result
        vanilla_search_chat_file = f"./user_intent_data/{dataset}/{chat_model}/{search_engine}/vanilla_search/{rerank_model}-{chat_model}-{dataset}-{split}.jsonl"
        data_lines = [json.loads(line) for line in open(vanilla_search_chat_file, "r", encoding="utf-8")]
    else:
        #  open the previous round result
        previous_round_chat_file = f"./user_intent_data/{dataset}/{chat_model}/intergen/{chat_model}round{round}chat-{dataset}-{split}.jsonl"
        data_lines = [json.loads(line) for line in open(previous_round_chat_file, "r", encoding="utf-8")]
    if args.mini_dataset:
        data_lines = data_lines[:10]

    data_lines = retrieval_round(data_lines)
    #  save the result
    with open(output_file, "w", encoding="utf-8") as f:
        for line in data_lines:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
