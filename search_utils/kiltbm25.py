import argparse
import json
import sys
import os
from time import sleep
import re

import requests

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from tqdm import tqdm


def bm25_search(queries, search_engine):
    patience = 3
    search_results = []
    while True:
        try:
            body = {'queries': queries}
            if search_engine == "kiltbm25":
                res = requests.request('GET', f'http://{address}/search/bm25/', json=body)
            else:
                raise NotImplementedError(f"search engine {search_engine} is not implemented")
            search_results = json.loads(res.text)["search_results"]
            for idx, result in enumerate(search_results):
                result['query'] = queries[idx]
            break
        except Exception as e:
            print(f"search query {queries} failed, error: {e}")
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
    arg_parser.add_argument("--search_engine", type=str, default="kiltbm25")
    arg_parser.add_argument("--address", type=str, default="172.17.66.79:5050")
    args = arg_parser.parse_args()
    dataset = args.dataset
    split = args.split
    search_method = args.search_method
    search_engine = args.search_engine
    address = args.address

    gpt4sep_base = "llama13b"
    sep_base = "llama7b"

    custom_rewrite_methods = ["llama_rewrite_search", "rule_rewrite_search"]

    if not os.path.exists(f"./user_intent_data/{dataset}/{search_engine}/{search_method}/"):
        os.makedirs(f"./user_intent_data/{dataset}/{search_engine}/{search_method}/", exist_ok=True)

    rewrite_method = ""  # question rewrite method
    sep_method = ""  # claim rewrite method
    if "plus" in search_method:
        #  use both question rewrite and claim rewrite, open both files
        rewrite_method = search_method.split("plus")[0]
        sep_method = search_method.split("sep_rewrite_search")[0].split("plus")[1]
        if "gpt4" in rewrite_method:
            with open(f"./user_intent_data/{dataset}/gpt4/gpt4-{dataset}-{split}.jsonl", "r", encoding="utf-8") as f:
                data_lines = [json.loads(line) for line in f]
        else:
            with open(
                    f"./user_intent_data/{dataset}/rewrite/{rewrite_method}/{rewrite_method}-{dataset}-{split}.jsonl",
                    "r", encoding="utf-8") as f:
                data_lines = [json.loads(line) for line in f]
        if "gpt4" in sep_method:
            #  temporarily use llama13b for claim rewrite
            with open(f"./user_intent_data/{dataset}/gpt4/gpt4sep-{gpt4sep_base}-{dataset}-{split}.jsonl", "r",
                      encoding="utf-8") as f:
                claim_data_lines = [json.loads(line) for line in f]
        else:
            #  temporarily use llama13b for claim rewrite
            with open(
                    f"./user_intent_data/{dataset}/rewrite/{sep_method}/{sep_method}sep-{sep_base}-{dataset}-{split}.jsonl",
                    "r", encoding="utf-8") as f:
                claim_data_lines = [json.loads(line) for line in f]
        data_lines = [{**data_lines[i], **claim_data_lines[i]} for i in range(len(data_lines))]

    elif search_method == "vanilla_search":
        with open(f"./user_intent_data/{dataset}/{dataset}-{split}.jsonl", "r", encoding="utf-8") as f:
            data_lines = [json.loads(line) for line in f]
    elif search_method == "gpt4_rewrite_search":
        with open(f"./user_intent_data/{dataset}/gpt4/gpt4-{dataset}-{split}.jsonl", "r", encoding="utf-8") as f:
            data_lines = [json.loads(line) for line in f]
    #  if search method is in the formate of "vxxxx_rewrite_search", or "vxxxxllama7b_rewrite_search"
    elif re.match(r"v\d[0-9a-z]+_rewrite_search", search_method) or search_method in custom_rewrite_methods:
        rewrite_model = search_method.replace("_rewrite_search", "")
        with open(f"./user_intent_data/{dataset}/rewrite/{rewrite_model}/{rewrite_model}-{dataset}-{split}.jsonl", "r",
                  encoding="utf-8") as f:
            data_lines = [json.loads(line) for line in f]
    else:
        raise NotImplementedError(f"search method {search_method} is not implemented")
    if args.mini_dataset:
        data_lines = data_lines[:20]
    #  take the first 5000 data
    data_lines =data_lines[:5000]
    print(f"{search_engine} search, dataset {dataset}, split {split}, search method {search_method}")
    output_file = f"./user_intent_data/{dataset}/{search_engine}/{search_method}/{search_engine}-{dataset}-{split}.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        pass
    for idx, data_line in tqdm(enumerate(data_lines), total=len(data_lines), desc=f"{search_engine}-{dataset}-{split}"):
        if "plus" in search_method:
            #  use both question rewrite and claim rewrite
            if "gpt4plus" in search_method:
                queries = [q["searchWord"] for q in data_line["questions"] if q["needSearch"]]
            else:
                queries = [q["searchWord"] for q in data_line[f"{rewrite_method}_rewrite"]["questions"] if
                           q["needSearch"]]

            if "plusgpt4sep" in search_method:
                queries += [q["query"] for q in data_line[f"{gpt4sep_base}_without_search_gpt4sep_claims"] if
                            q["needSearch"]]
            else:
                queries += [q["query"] for q in data_line[f"{sep_base}_without_search_{sep_method}sep_claims"] if
                            q["needSearch"]]
            search_word_results = []
            if idx == 0:
                print(queries)
            try:
                search_word_results = bm25_search(queries, search_engine)
            except:
                print(f"error: {idx}, {queries}")
                search_word_results = []
            data_lines[idx][f"{search_method}_results"] = search_word_results

        elif search_method == "vanilla_search":
            #  do search with vanilla query
            query = data_line["question"]
            if idx == 0:
                print(query)
            try:
                vanilla_query_results = bm25_search([query], search_engine)[0]
            except:
                print(f"error: {idx}, {query}")
                vanilla_query_results = []
            data_lines[idx]["vanilla_search_results"] = vanilla_query_results
        elif search_method == "gpt4_rewrite_search":
            #  do search with rewritten query
            queries = [data_line["question"]]
            queries += [q["searchWord"] for q in data_line["questions"] if q["needSearch"]]
            search_word_results = []
            if idx == 0:
                print(queries)
            try:
                search_word_results = bm25_search(queries, search_engine)
            except:
                print(f"error: {idx}, {queries}")
                search_word_results = []
            data_lines[idx]["gpt4_rewrite_search_results"] = search_word_results
        elif search_method in custom_rewrite_methods or re.match(r"v\d[0-9a-z]+_rewrite_search", search_method):
            #  do search with rewritten query
            queries = [data_line["question"]]
            if "claims" in data_line[f"{search_method.replace('_search', '')}"] and len(
                    data_line[f"{search_method.replace('_search', '')}"]["claims"]) > 0:
                if idx == 0:
                    print("search with claims and questions")
                queries += [q["query"] for q in data_line[f"{search_method.replace('_search', '')}"]["questions"] if
                            q["needSearch"]]
                if "known" in data_line[f"{search_method.replace('_search', '')}"]["claims"][0]:
                    # if idx == 0:
                    #     print(f"{idx} search with known claims")
                    queries += [q["query"] for q in data_line[f"{search_method.replace('_search', '')}"]["claims"] if
                                not q["known"] and "query" in q and q["query"] != ""]
                    # if idx == 0:
                    #     print(data_line[f"{search_method.replace('_search', '')}"]["claims"])
                    #     print([q["query"] for q in data_line[f"{search_method.replace('_search', '')}"]["claims"] if
                    #             not q["known"] and "query" in q and q["query"] != ""])
                else:
                    #  search every claim induced query
                    queries += [q["query"] for q in data_line[f"{search_method.replace('_search', '')}"]["claims"] if
                                q["needSearch"] and "query" in q and q["query"] != ""]
            else:
                if idx == 0:
                    print("search with questions")
                queries += [q["query"] for q in data_line[f"{search_method.replace('_search', '')}"]["questions"]
                            if q["needSearch"]]

            search_word_results = []
            if idx == 2:
                print(queries)
            try:
                search_word_results = bm25_search(queries, search_engine)
            except:
                print(f"error: {idx}, {queries}")
                search_word_results = []
            data_lines[idx][f"{search_method}_results"] = search_word_results
        else:
            raise NotImplementedError(f"search method {search_method} is not implemented")
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(data_lines[idx], ensure_ascii=False) + "\n")
