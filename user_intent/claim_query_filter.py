import argparse
import re

import requests
import json
import os
import concurrent.futures

import torch
from tqdm import tqdm

params_query_rewrite = {"repetition_penalty": 1.05, "temperature": 0.01, "top_k": 1, "top_p": 0.85,
                        "max_new_tokens": 512,
                        "do_sample": False, "seed": 2023}
headers = {'Content-Type': 'application/json'}
WIKI_NUM_PASSAGES = 20
BING_NUM_SNIPPETS = 20

def sending_request(address, question):
    if local_model is not None and args.local_inference:
        input_ids = local_tokenizer.encode(question, return_tensors="pt")
        len_input_ids = len(input_ids[0])
        if torch.cuda.is_available():
            input_ids = input_ids.cuda()
        outputs = local_model.generate(input_ids)
        res = local_tokenizer.decode(outputs[0][len_input_ids:], skip_special_tokens=True)
        return res
    else:
        data = {"inputs": question, "parameters": params_query_rewrite}
        request = requests.post(f"http://{address}", json=data, headers=headers)
        return json.loads(request.text)[0]['generated_text']

def construct_input_text(json_line):
    prompt = (f"<s>[INST] <<SYS>>\nYou are a helpful assistant. Your task is to parse user input into"
              f" structured formats according to the coarse answer. Current datatime is 2023-12-20 9:47:28"
              f" <</SYS>>\n Course answer: (({json_line['claim']}))\nQuestion: "
              f"(({json_line['query']})) [/INST]")

    return prompt

if __name__ == '__main__':
    #  init 4 threads
    #  each thread has a session
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--dataset", type=str, default="webglm-qa")
    arg_parser.add_argument("--split", type=str, default="test")
    arg_parser.add_argument("--search_method", type=str, default="vanilla_search")
    arg_parser.add_argument("--mini_dataset", action="store_true", help="whether to use mini dataset")
    arg_parser.add_argument("--judge_model", type=str, default="llama13b")
    arg_parser.add_argument("--inference_url", type=str, default="")
    arg_parser.add_argument("--local_inference", action="store_true")
    args = arg_parser.parse_args()
    dataset = args.dataset
    split = args.split
    search_method = args.search_method
    judge_model = args.judge_model
    INFERENCE_URL = args.inference_url


    if re.match(r"v\d[0-9a-z]+_rewrite_search", search_method):
        rewrite_model = search_method.replace("_rewrite_search", "")
        if not os.path.exists(f"./user_intent_data/{dataset}/rewrite/{rewrite_model}{judge_model}/"):
            os.makedirs(f"./user_intent_data/{dataset}/rewrite/{rewrite_model}{judge_model}/", exist_ok=True)
        input_file = f"./user_intent_data/{dataset}/rewrite/{rewrite_model}/{rewrite_model}-{dataset}-{split}.jsonl"
        output_file = f"./user_intent_data/{dataset}/rewrite/{rewrite_model}{judge_model}/{rewrite_model}{judge_model}-{dataset}-{split}.jsonl"
        with open(input_file, "r", encoding="utf-8") as f:
            data_lines = [json.loads(line) for line in f]

    else:
        raise NotImplementedError(f"search method {search_method} is not implemented")

    print(f"INFERENCE_URL: {INFERENCE_URL}")

    with open(output_file, "w", encoding="utf-8") as f:
        pass
    with open(input_file, "r", encoding="utf-8") as f:
        data_lines = [json.loads(line) for line in f]

    local_model = None
    if args.local_inference:
        model_path = [file for file in os.listdir("./models") if judge_model in file][0]
        model_path = os.path.join("./models", model_path)
        print(f"local model path: {model_path}")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        local_model = AutoModelForCausalLM.from_pretrained(model_path).eval()
        if torch.cuda.is_available():
            local_model.cuda()
        local_tokenizer = AutoTokenizer.from_pretrained(model_path)


    if args.mini_dataset:
        data_lines = data_lines[:20]
    #  trim dataset that is too large
    data_lines = data_lines[:5000]
    print(
        f"dataset: {dataset}, split: {split}, search method: {search_method}, judge model: {judge_model}, inference url: {INFERENCE_URL}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        for idx, data_line in tqdm(enumerate(data_lines), total=len(data_lines),
                                   desc=f"dataset: {dataset}, split: {split}"):
            filtered_claims = []
            for claim in data_line[f"{search_method.replace('_search', '')}"]["claims"]:
                if not claim["query"]:
                    claim["known"] = False
                    filtered_claims.append(claim)
                    continue
                input_text = construct_input_text(claim)
                if idx == 0:
                    print(f"input text: {input_text}")
                future_res = executor.submit(sending_request, INFERENCE_URL, input_text)
                output_str = future_res.result()
                try:
                    known = output_str.split("<Known(")[1].split(")>")[0]
                    known = True if known == "True" else False
                except:
                    print(f"parse known error: {output_str}")
                    known = False
                claim["known"] = known
                filtered_claims.append(claim)
            data_lines[idx][f"{rewrite_model}{judge_model}_rewrite"] = data_lines[idx][f"{search_method.replace('_search', '')}"]
            data_lines[idx].pop(f"{search_method.replace('_search', '')}")

            data_lines[idx][f"{rewrite_model}{judge_model}_rewrite"]["claims"] = filtered_claims
        with open(output_file, "a", encoding="utf-8") as f:
            #  write to file
            for data_line in data_lines:
                f.write(json.dumps(data_line, ensure_ascii=False) + "\n")