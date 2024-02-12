import argparse
import json
import sys
import os
from time import sleep

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from tqdm import tqdm
from search_utils.bing_search import BingSearch

def stable_search(query):
    patience = 10
    search_results = []
    while True:
        try:
            search_results = BingSearch.search(query)
            break
        except Exception as e:
            print(f"search query {query} failed, error: {e}")
            patience -= 1
            sleep(1)
            if patience <= 0:
                break
    return search_results

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--dataset", type=str, default="atis", help="dataset name")
    arg_parser.add_argument("--split", type=str, default="test", help="split name")
    arg_parser.add_argument("--search_method", type=str, default="vanilla_search", help="search method")
    arg_parser.add_argument("--mini_dataset", action="store_true", help="whether to use mini dataset")
    args = arg_parser.parse_args()
    dataset = args.dataset
    split = args.split
    search_method = args.search_method

    if not os.path.exists(f"./user_intent_data/{dataset}/bing/{search_method}/"):
        os.makedirs(f"./user_intent_data/{dataset}/bing/{search_method}/", exist_ok=True)

    if search_method == "vanilla_search":
        with open(f"./user_intent_data/{dataset}/{dataset}-{split}.jsonl", "r", encoding="utf-8") as f:
            data_lines = [json.loads(line) for line in f]
    elif search_method == "gpt4_rewrite_search":
        with open(f"./user_intent_data/{dataset}/gpt4/gpt4-{dataset}-{split}.jsonl", "r", encoding="utf-8") as f:
            data_lines = [json.loads(line) for line in f]
    elif search_method == "llama_rewrite_search":
        with open(f"./user_intent_data/{dataset}/llama_rewrite/llama_rewrite-{dataset}-{split}.jsonl", "r", encoding="utf-8") as f:
            data_lines = [json.loads(line) for line in f]
    else:
        raise NotImplementedError(f"search method {search_method} is not implemented")
    if args.mini_dataset:
        data_lines = data_lines[:10]
    print(f"bing search dataset {dataset}, split {split}, search method {search_method}")

    with open(f"./user_intent_data/{dataset}/bing/{search_method}/bing-{dataset}-{split}.jsonl", "w",
              encoding="utf-8") as f:
        pass
    for idx, data_line in tqdm(enumerate(data_lines), total=len(data_lines), desc=f"bing-{dataset}-{split}"):
        if search_method == "vanilla_search":
            #  do search with vanilla query
            query = data_line["question"]
            if idx == 0:
                print(query)
            try:
                vanilla_query_results = stable_search(query)
            except:
                print(f"error: {idx}, {query}")
                vanilla_query_results = []
            data_lines[idx]["vanilla_search_results"] = vanilla_query_results
        elif search_method == "gpt4_rewrite_search":
            #  do search with rewritten query
            queries=[q["searchWord"] for q in data_line["questions"] if q["needSearch"]]
            search_word_results = []
            if idx == 0:
                print(queries)
            for query in queries:
                try:
                    search_word_results.append(stable_search(query))
                except:
                    print(f"error: {idx}, {query}")
                    search_word_results.append([])
            data_lines[idx]["gpt4_rewrite_search_results"] = search_word_results
        elif search_method == "llama_rewrite_search":
            #  do search with rewritten query
            queries=[q["searchWord"] for q in data_line["llama_rewrite"]["questions"] if q["needSearch"]]
            search_word_results = []
            if idx == 0:
                print(queries)
            for query in queries:
                try:
                    search_word_results.append(stable_search(query))
                except:
                    print(f"error: {idx}, {query}")
                    search_word_results.append([])
            data_lines[idx]["llama_search_word_results"] = search_word_results
        else:
            raise NotImplementedError(f"search method {search_method} is not implemented")
        with open(f"./user_intent_data/{dataset}/bing/{search_method}/bing-{dataset}-{split}.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(data_lines[idx], ensure_ascii=False) + "\n")
