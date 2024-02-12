#  filter out known/unknown results according to long answers only based on no search results
datasets = ['eli5']
split = ('train')
chat_model = "qwen7b"
search_engine = "kiltbm25"
rerank_model = "e5base"

import string
import re
import evaluate
import random
import json
import os
import numpy as np
import tqdm

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
    for idx in tqdm.tqdm(range(len(generated_answers)), desc=f"compute rouge for {len(generated_answers)} samples"):
        rouge_results.append(calculate_one_rouge_score(generated_answers[idx], gold_answers[idx]))
    # print(rouge_results)
    return rouge_results


for dataset in datasets:
    # print(f"dataset: {dataset}, split: {split}, chat_model: {chat_model}, search_engine: {search_engine}, rerank_model: {rerank_model}")
    #  no search chat results
    no_search_chat_results_file = f'./user_intent_data/{dataset}/{chat_model}/without_search/{chat_model}-{dataset}-{split}.jsonl'
    no_search_chat_results = [json.loads(line) for line in open(no_search_chat_results_file, "r", encoding="utf-8")]
    no_search_generated_answers = [data_line[f"{chat_model}_without_search_answer"] for data_line in
                                   no_search_chat_results]

    if "answer" in no_search_chat_results[0]:
        gold_answers = [data_line["answer"] for data_line in no_search_chat_results]
    else:
        gold_answers = [data_line["long_answers"] for data_line in no_search_chat_results]

    no_search_rouge_result = compute_rouge_for_each(no_search_generated_answers, gold_answers)
    for idx in range(len(no_search_rouge_result)):
        no_search_chat_results[idx]["rouge"] = no_search_rouge_result[idx]

        #  rank the rouge results, take the top 20% as known, the rest as unknown
    ranked_samples = sorted(no_search_chat_results, key=lambda x: x["rouge"]["rougeL"], reverse=True)
    known_samples, unknown_samples = (ranked_samples[:int(len(ranked_samples) * 0.2)],
                                      ranked_samples[int(len(ranked_samples) * 0.2):])
    #  add known/unknown tags
    for idx in range(len(known_samples)):
        known_samples[idx]["known"] = True
    for idx in range(len(unknown_samples)):
        unknown_samples[idx]["known"] = False

    #  sample unknown results
    unknown_rouge_results = random.sample(unknown_samples, len(known_samples))

    filtered_samples = random.sample(known_samples + unknown_rouge_results, len(known_samples + unknown_rouge_results))

    #  save known/unknown samples
    print(f"filtered {len(known_samples)} known samples and {len(unknown_rouge_results)} unknown samples")
    print(
        f"dataset: {dataset}, split: {split}, chat_model: {chat_model}, search_engine: {search_engine}, rerank_model: {rerank_model}")
    if not os.path.exists(f"./user_intent_data/mixed/{dataset}/{chat_model}/{search_engine}/"):
        os.makedirs(f"./user_intent_data/mixed/{dataset}/{chat_model}/{search_engine}/", exist_ok=True)
    #  save filtered results to mixed dataset
    mixed_dataset_file = f"./user_intent_data/mixed/{dataset}/{chat_model}/{search_engine}/filtered-{dataset}-{split}.jsonl"
    with open(mixed_dataset_file, "w", encoding="utf-8") as f:
        for idx in range(len(filtered_samples)):
            f.write(json.dumps(filtered_samples[idx], ensure_ascii=False) + "\n")
