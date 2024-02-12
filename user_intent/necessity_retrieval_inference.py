import argparse
import json
import os
import re

import torch
from tqdm import tqdm
from transformers import LlamaForCausalLM, LlamaTokenizer


def construct_input_text(json_line, language):
    if language == "zh":
        prompt = f"<s>[INST] <<SYS>>\n你是一个有用的助手。你的任务是将用户输入解析为结构化格式。当前时间是2023-11-20 9:47:28 <</SYS>>\n{json_line['question']} [/INST]"
    elif language == "en":
        prompt = f"<s>[INST] <<SYS>>\nYou are a helpful assistant. Your task is to parse user input into structured formats. Current datatime is 2023-11-20 9:47:28 <</SYS>>\n{json_line['question']} [/INST]"
        if provide_without_search_answer:
            prompt = (f"<s>[INST] <<SYS>>\nYou are a helpful assistant. Your task is to parse user input into"
                      f" structured formats according to the coarse answer. Current datatime is 2023-12-20 9:47:28"
                      f" <</SYS>>\n Course answer: (({json_line[key_without_search_answer]}))\nQuestion: "
                      f"(({json_line['question']})) [/INST]")
    else:
        raise ValueError(f"language: {language} not supported")
    return prompt


def cal_ppl(query, answer):
    input_ids = tokenizer.encode(query, add_special_tokens=False)
    target_ids = tokenizer.encode(answer, add_special_tokens=False)
    input_tensor = torch.tensor([input_ids + target_ids]).to(model.device)
    target_tensor = torch.tensor([[-100] * len(input_ids) + target_ids]).to(model.device)
    with torch.no_grad():
        outputs = model(input_tensor, labels=target_tensor)

        # loss is calculated using CrossEntropyLoss which averages over valid labe
        # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
        # to the left by 1.
        neg_log_likelihood = outputs.loss
        # print(neg_log_likelihood)
        # nlls.append(neg_log_likelihood)
        ppl = neg_log_likelihood.to(torch.float32).detach().cpu().item()
        # ppl = torch.exp(torch.stack(nlls).mean()).to(torch.float32).detach().cpu().numpy()
    return ppl


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--dataset", type=str, default="webglm-qa")
    arg_parser.add_argument("--split", type=str, default="test")
    arg_parser.add_argument("--mini_dataset", action="store_true", help="whether to use mini dataset")
    arg_parser.add_argument("--answer_model", type=str, default="llama13b")
    arg_parser.add_argument("--provide_without_search_answer", action="store_true")
    arg_parser.add_argument("--judge_model", type=str, default="")
    args = arg_parser.parse_args()
    dataset = args.dataset
    split = args.split
    answer_model = args.answer_model
    provide_without_search_answer = args.provide_without_search_answer
    judge_model = args.judge_model

    if re.match(r"v\d+", judge_model):
        #  find model path from ./models
        model_path = [file for file in os.listdir("./models") if judge_model in file][0]
        model_path = os.path.join("./models", model_path)
        model = LlamaForCausalLM.from_pretrained(model_path)
        if torch.cuda.is_available():
            model = model.cuda()
        model.eval()
        tokenizer = LlamaTokenizer.from_pretrained(model_path)
    else:
        raise NotImplementedError(f"judge model {judge_model} not implemented")

    # if dataset is in forms like vxxxxx
    if re.match(r"v\d+", dataset):
        if not os.path.exists(f"./user_intent_data/mixed/{dataset}/rewrite/{judge_model}"):
            os.makedirs(f"./user_intent_data/mixed/{dataset}/rewrite/{judge_model}", exist_ok=True)
        file_dir = f"./user_intent_data/mixed/{dataset}"
        split_file = [file for file in os.listdir(file_dir) if split in file][0]
        input_file = os.path.join(file_dir, split_file)
        output_file = f"./user_intent_data/mixed/{dataset}/rewrite/{judge_model}/{judge_model}ppl-{dataset}-{split}.jsonl"
    else:
        if provide_without_search_answer:
            #  the result of rewrite model is influenced by the answer model
            judge_model += answer_model
            input_file = f"./user_intent_data/{dataset}/{answer_model}/without_search/{answer_model}-{dataset}-{split}.jsonl"
        else:
            input_file = f"./user_intent_data/{dataset}/gpt4/gpt4-{dataset}-{split}.jsonl"

        if not os.path.exists(f"./user_intent_data/{dataset}/rewrite/{judge_model}"):
            os.makedirs(f"./user_intent_data/{dataset}/rewrite/{judge_model}", exist_ok=True)
        output_file = f"./user_intent_data/{dataset}/rewrite/{judge_model}/{judge_model}ppl-{dataset}-{split}.jsonl"

    with open(output_file, "w", encoding="utf-8") as f:
        pass
    with open(input_file, "r", encoding="utf-8") as f:
        data_lines = [json.loads(line) for line in f]

    if provide_without_search_answer:
        #  find key name ending with _without_search_answer
        try:
            key_without_search_answer = [key for key in data_lines[0].keys() if key.endswith("_without_search_answer")][
                0]
            print(f"key without search answer: {key_without_search_answer}")
        except IndexError:
            print(f"key without search answer not found in data_lines[0]: {data_lines[0]}")
            print("continue with no without search answer")
            provide_without_search_answer = False

    if args.mini_dataset:
        data_lines = data_lines[:20]
    #  trim dataset that is too large
    data_lines = data_lines[:5000]
    print(f"dataset: {dataset}, split: {split}, answer_model: {answer_model}, judge_model: {judge_model}")

    for idx, data_line in tqdm(enumerate(data_lines), total=len(data_lines),
                               desc=f"dataset: {dataset}, split: {split}"):
        #  do search with vanilla query
        if provide_without_search_answer:
            model_input = construct_input_text(data_line, language="en")
        else:
            model_input = "<s>[INST] " + data_line["question"] + " [/INST]"
        if idx == 0:
            print(f"model input: {model_input}")

        #  perplexity inference
        known_ppl = cal_ppl(model_input, "<Known(True)>")
        unknown_ppl = cal_ppl(model_input, "<Known(False)>")

        data_line["known_ppl"] = known_ppl - unknown_ppl
    with open(output_file, "a", encoding="utf-8") as f:
        for data_line in data_lines:
            f.write(json.dumps(data_line) + "\n")
