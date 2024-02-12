import argparse

import requests
import json
import os
import concurrent.futures
import re
from tqdm import tqdm

params_query_rewrite = {"repetition_penalty": 1.05, "temperature": 0.01, "top_k": 1, "top_p": 0.85,
                        "max_new_tokens": 512,
                        "do_sample": False, "seed": 2023}
headers = {'Content-Type': 'application/json'}


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


def construct_input_text(json_line, language):
    if language == "zh":
        prompt = f"<s>[INST] <<SYS>>\n你是一个有用的助手。你的任务是将用户输入解析为结构化格式。当前时间是2023-11-20 9:47:28 <</SYS>>\n{json_line['question']} [/INST]"
    elif language == "en":
        # if "rewrite" in model_path:
        if provide_without_search_answer:
            prompt = (f"<s>[INST] <<SYS>>\nYou are a helpful assistant. Your task is to parse user input into"
                      f" structured formats according to the coarse answer. Current datatime is 2023-12-20 9:47:28"
                      f" <</SYS>>\n Course answer: (({json_line[key_without_search_answer]}))\nQuestion: "
                      f"(({json_line['question']})) [/INST]")
        else:
            prompt = (f"<s>[INST] <<SYS>>\nYou are a helpful assistant. Your task is to parse user input into"
                      f" structured formats. Current datatime is 2023-12-20 9:47:28"
                      f" <</SYS>>\n{json_line['question']} [/INST]")
        # elif "search-tag" in model_path:
        #     if provide_without_search_answer:
        #         prompt = (f"<s>[INST] <<SYS>>\nYou are a helpful assistant. Your task is to judge whether the model"
        #                  f" has known the information about the question according to the coarse answer."
        #                  f" <</SYS>>\n Course answer: (({json_line[key_without_search_answer]}))\nQuestion: "
        #                  f"(({json_line['question']})) [/INST]")
        #     else:
        #         prompt = (f"<s>[INST] <<SYS>>\nYou are a helpful assistant. Your task is to parse user input into"
        #                  f" structured formats. Current datatime is 2023-12-20 9:47:28"
        #                  f" <</SYS>>\n{json_line['question']} [/INST]")
        # else:
        #     raise ValueError(f"task_name {model_path} is not supported")

    else:
        raise ValueError(f"language: {language} not supported")
    return prompt


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--dataset", type=str, default="webglm-qa")
    arg_parser.add_argument("--split", type=str, default="test")
    arg_parser.add_argument("--language", type=str, default="zh")
    arg_parser.add_argument("--address", type=str, default="")
    arg_parser.add_argument("--multiturn", action="store_true")
    arg_parser.add_argument("--rewrite_model", type=str, default="")
    arg_parser.add_argument("--answer_model", type=str, default="")
    arg_parser.add_argument("--mini_dataset", action="store_true")
    arg_parser.add_argument("--provide_without_search_answer", action="store_true")
    arg_parser.add_argument("--local_inference", action="store_true")
    args = arg_parser.parse_args()
    dataset = args.dataset
    split = args.split
    language = args.language
    address = args.address
    multiturn = args.multiturn
    rewrite_model = args.rewrite_model
    answer_model = args.answer_model
    provide_without_search_answer = args.provide_without_search_answer

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


    # if dataset is in forms like vxxxxx
    if re.match(r"v\d+", dataset):
        if not os.path.exists(f"./user_intent_data/mixed/{dataset}/rewrite/{rewrite_model}"):
            os.makedirs(f"./user_intent_data/mixed/{dataset}/rewrite/{rewrite_model}", exist_ok=True)
        file_dir = f"./user_intent_data/mixed/{dataset}"
        split_file = [file for file in os.listdir(file_dir) if split in file][0]
        input_file = os.path.join(file_dir, split_file)
        output_file = f"./user_intent_data/mixed/{dataset}/rewrite/{rewrite_model}/unparsed-{rewrite_model}-{dataset}-{split}.jsonl"
    else:
        if provide_without_search_answer:
            #  the result of rewrite model is influenced by the answer model
            rewrite_model += answer_model
            input_file = f"./user_intent_data/{dataset}/{answer_model}/without_search/{answer_model}-{dataset}-{split}.jsonl"
        else:
            input_file = f"./user_intent_data/{dataset}/gpt4/gpt4-{dataset}-{split}.jsonl"

        if not os.path.exists(f"./user_intent_data/{dataset}/rewrite/{rewrite_model}"):
            os.makedirs(f"./user_intent_data/{dataset}/rewrite/{rewrite_model}", exist_ok=True)
        output_file = f"./user_intent_data/{dataset}/rewrite/{rewrite_model}/unparsed-{rewrite_model}-{dataset}-{split}.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        pass
    with open(input_file, "r", encoding="utf-8") as f:
        data_lines = [json.loads(line) for line in f]
    if args.mini_dataset:
        data_lines = data_lines[:20]
    data_lines = data_lines[:5000]
    print(
        f"dataset: {dataset}, split: {split}, language: {language}, port: {address}, multiturn: {multiturn}, rewrite_model: {rewrite_model}")
    if provide_without_search_answer:
        #  find key name ending with _without_search_answer
        try:
            key_without_search_answer = [key for key in data_lines[0].keys() if key.endswith("_without_search_answer")][
                0]
            print(f"key without search answer: {key_without_search_answer}")
            assert answer_model in key_without_search_answer, f"answer model: {answer_model}, key: {key_without_search_answer} is not matched"
        except IndexError:
            print(f"key without search answer not found in data_lines[0]: {data_lines[0]}")
            print("continue with no without search answer")
            provide_without_search_answer = False

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        for idx, data_line in tqdm(enumerate(data_lines), total=len(data_lines),
                                   desc=f"dataset: {dataset}, split: {split}"):
            if multiturn:
                history = ""
                current_time = "2023-11-20 9:47:28"
                for conversation in data_line["conversations"]:
                    if conversation["from"] == "human":
                        assert "<C_Q>" not in history, f"history: {history}"
                        history += "<C_Q>" + conversation["value"]
                        inputs = history + "<NOW>" + current_time + "<Pred>"
                        history = history.replace("<C_Q>", "<H_Q>")
                        # print(inputs)
                        assert len(inputs.split("<C_Q>")) == 2, f"inputs: {inputs}"
                        future_res = executor.submit(sending_request, address, inputs)
                        res = future_res.result()
                        conversation[f"{rewrite_model}_rewrite"] = res

                    elif conversation["from"] == "gpt":
                        history += "<H_A>" + conversation["value"]

            else:
                input_text = construct_input_text(data_line, language)
                if idx == 0:
                    print(f"input text: {input_text}")
                future_res = executor.submit(sending_request, address, input_text)
                res = future_res.result()
                data_lines[idx][f"{rewrite_model}_rewrite"] = res
        with open(output_file, "a", encoding="utf-8") as f:
            #  write to file
            for data_line in data_lines:
                f.write(json.dumps(data_line, ensure_ascii=False) + "\n")
