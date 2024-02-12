import argparse
import json
import os
import random

import evaluate
from tqdm import tqdm

random.seed(4545)

rouge = evaluate.load("./evaluate_utils/rouge/")

def calculate_one_rouge_score(generated_answer, gold_answers):
    max_rouge_score = {"rouge1": 0, "rouge2": 0, "rougeL": 0}
    candidate_idx = -1
    if isinstance(gold_answers, str):
        return rouge.compute(predictions=[generated_answer], references=[gold_answers])
    if len(gold_answers) == 1:
        return rouge.compute(predictions=[generated_answer], references=[gold_answers[0]])
    for idx, gold_answer in enumerate(gold_answers):
        rouge_score = rouge.compute(predictions=[generated_answer], references=[gold_answer])
        if rouge_score["rougeL"] > max_rouge_score["rougeL"]:
            max_rouge_score = rouge_score
            candidate_idx = idx
    return max_rouge_score


def compute_rouge_for_each(generated_answers, gold_answers):
    rouge_results = []
    for idx in tqdm(range(len(generated_answers)), desc=f"compute rouge for {len(generated_answers)} samples"):
        rouge_results.append(calculate_one_rouge_score(generated_answers[idx], gold_answers[idx]))
    # print(rouge_results)
    return rouge_results

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--dataset", type=str, default="webglm-qa")
    arg_parser.add_argument("--split", type=str, default="test")
    arg_parser.add_argument("--mini_dataset", action="store_true", help="whether to use mini dataset")
    arg_parser.add_argument("--chat_model", type=str, default="llama13b")
    arg_parser.add_argument("--search_engine", type=str, default="kiltbm25")
    arg_parser.add_argument("--rerank_model", type=str, default="e5base")
    arg_parser.add_argument("--prompt_method", type=str, default="without_search")
    args = arg_parser.parse_args()
    dataset = args.dataset
    split = args.split
    prompt_method = args.prompt_method
    chat_model = args.chat_model
    search_engine = args.search_engine
    rerank_model = args.rerank_model

    if "without_search" in prompt_method:
        if not os.path.exists(f"./user_intent_data/{dataset}/{chat_model}/{prompt_method}/"):
            os.makedirs(f"./user_intent_data/{dataset}/{chat_model}/{prompt_method}/", exist_ok=True)
        input_file = f'./user_intent_data/{dataset}/{chat_model}/{prompt_method}/{chat_model}-{dataset}-{split}.jsonl'
        output_file = f'./user_intent_data/{dataset}/{chat_model}/{prompt_method}/{chat_model}rouge-{dataset}-{split}.jsonl'
    else:
        if not os.path.exists(f"./user_intent_data/{dataset}/{chat_model}/{search_engine}/{prompt_method}/"):
            os.makedirs(f"./user_intent_data/{dataset}/{chat_model}/{search_engine}/{prompt_method}/", exist_ok=True)
        if "bing" in search_engine:
            input_file = f"./user_intent_data/{dataset}/{chat_model}/{search_engine}/{prompt_method}/{chat_model}-{dataset}-{split}.jsonl"
            output_file = f"./user_intent_data/{dataset}/{chat_model}/{search_engine}/{prompt_method}/{chat_model}rouge-{dataset}-{split}.jsonl"
        else:
            input_file = f"./user_intent_data/{dataset}/{chat_model}/{search_engine}/{prompt_method}/{rerank_model}-{chat_model}-{dataset}-{split}.jsonl"
            output_file = f"./user_intent_data/{dataset}/{chat_model}/{search_engine}/{prompt_method}/{rerank_model}-{chat_model}rouge-{dataset}-{split}.jsonl"

    with open(output_file, "w", encoding="utf-8") as f:
        pass
    with open(input_file, "r", encoding="utf-8") as f:
        data_lines = [json.loads(line) for line in f]

    if args.mini_dataset:
        data_lines = data_lines[:20]
    #  trim dataset that is too large
    no_search_chat_results = data_lines[:5000]
    print(f"dataset: {dataset}, split: {split}, prompt_method: {prompt_method}, chat_model: {chat_model}, search_engine: {search_engine}, rerank_model: {rerank_model}")

    no_search_generated_answers = [data_line[f"{chat_model}_{prompt_method}_answer"] for data_line in
                                   no_search_chat_results]
    if "answer" in no_search_chat_results[0]:
        gold_answers = [data_line["answer"] for data_line in no_search_chat_results]
    else:
        gold_answers = [data_line["long_answers"] for data_line in no_search_chat_results]

    no_search_rouge_result = compute_rouge_for_each(no_search_generated_answers, gold_answers)
    for idx in range(len(no_search_rouge_result)):
        no_search_chat_results[idx]["rouge"] = no_search_rouge_result[idx]
    with open(output_file, "a", encoding="utf-8") as f:
        for data_line in data_lines:
            f.write(json.dumps(data_line) + "\n")
