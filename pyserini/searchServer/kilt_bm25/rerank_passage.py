import argparse
import json
import os

import torch
import torch.nn.functional as F

from torch import Tensor
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel


def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


tokenizer = AutoTokenizer.from_pretrained('../../../../huggingface/e5-base-v2')
model = AutoModel.from_pretrained('../../../../huggingface/e5-base-v2')
if torch.cuda.is_available():
    model.to("cuda")
model.eval()
TOP_K = 20
MAX_BATCH_SIZE = 64


def calculate_rerank_score(query, passages):
    # Each input text should start with "query: " or "passage: ".
    # For tasks other than retrieval, you can simply use the "query: " prefix.

    input_texts = [query] + passages
    #  tokenize the input texts
    batch_dict = tokenizer(input_texts, max_length=512, padding=True, truncation=True, return_tensors='pt')
    if torch.cuda.is_available():
        batch_dict = {k: v.to("cuda") for k, v in batch_dict.items()}
    #  if batch is too large, split it into smaller batches
    if batch_dict['input_ids'].shape[0] > MAX_BATCH_SIZE:
        outputs = []
        for i in range(0, batch_dict['input_ids'].shape[0], 16):
            outputs.append(model(**{k: v[i:i + 16] for k, v in batch_dict.items()}).last_hidden_state.detach().cpu())
            # print(f"output shape: {outputs[-1].last_hidden_state.shape}")
        outputs = torch.cat([output for output in outputs], dim=0)
        # print(f"outputs shape: {outputs.shape}")
        embeddings = average_pool(outputs, batch_dict['attention_mask'].detach().cpu())
    else:
        outputs = model(**batch_dict)
        embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask']).detach().cpu()
    #  normalize embeddings
    embeddings = F.normalize(embeddings, p=2, dim=1)
    scores = (embeddings[0] @ embeddings[1:].T) * 100
    return scores


def rerank_passage(query, docs):
    search_results = []
    query = "query: " + query
    if isinstance(docs, dict):
        #  single query result
        docs = docs["docs"]
    elif isinstance(docs, list):
        #  multiple query results
        #  remove docs with same doc id
        doc_ids = set()
        unique_docs = []
        for _docs in docs:
            for idx, doc in enumerate(_docs["docs"]):
                if _docs["doc_ids"][idx] in doc_ids:
                    continue
                doc_ids.add(_docs["doc_ids"][idx])
                unique_docs.append(doc)
        docs = unique_docs

    else:
        raise NotImplementedError(f"docs type {type(docs)} is not implemented")
    for idx, doc in enumerate(docs):
        #  remove passages starts with "BULLET::::" or "Section::::"
        # skip_prefix = ["BULLET::::", "Section::::"]
        passages = ["passage: " + passage for passage in doc if
                    not passage.startswith("BULLET::::") and not passage.startswith("Section::::")]
        scores = calculate_rerank_score(query, passages)
        ranked_doc = [passages[i.tolist()][len("passage: "):] for i in scores.argsort(descending=True)[:TOP_K]]
        docs[idx] = ranked_doc
    #  take top 10 passage from each doc
    #  switch columns and rows
    docs = list(zip(*docs))
    #  flatten the list
    docs = [item for sublist in docs for item in sublist]
    search_results = docs[:TOP_K]

    return search_results


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--dataset", type=str, default="atis", help="dataset name")
    arg_parser.add_argument("--split", type=str, default="test", help="split name")
    arg_parser.add_argument("--search_method", type=str, default="vanilla_search", help="search method")
    arg_parser.add_argument("--mini_dataset", action="store_true", help="whether to use mini dataset")
    arg_parser.add_argument("--search_engine", type=str, default="kiltbm25")
    arg_parser.add_argument("--rerank_model", type=str, default="e5base")
    args = arg_parser.parse_args()
    dataset = args.dataset
    split = args.split
    search_method = args.search_method
    search_engine = args.search_engine
    rerank_model = args.rerank_model

    if not os.path.exists(f"./user_intent_data/{dataset}/{search_engine}/{search_method}/"):
        os.makedirs(f"./user_intent_data/{dataset}/{search_engine}/{search_method}/", exist_ok=True)

    with open(f"./user_intent_data/{dataset}/{search_engine}/{search_method}/{search_engine}-{dataset}-{split}.jsonl",
              "r", encoding="utf-8") as f:
        data_lines = [json.loads(line) for line in f]
    if args.mini_dataset:
        data_lines = data_lines[:10]
    #  take first 5000 data lines
    data_lines = data_lines[:5000]
    with open(
            f"./user_intent_data/{dataset}/{search_engine}/{search_method}/{rerank_model}-{search_engine}-{dataset}-{split}.jsonl",
            "w", encoding="utf-8") as f:
        pass

    print(
        f"{rerank_model} rerank, search engine {search_engine}, dataset {dataset}, split {split}, search method {search_method}")

    for idx, data_line in tqdm(enumerate(data_lines), total=len(data_lines),
                               desc=f"{rerank_model}-{search_engine}-{dataset}-{split}"):
        query = data_line["question"]
        if idx == 0:
            print(query)
        rerank_results = rerank_passage(data_line, search_method)
        data_lines[idx][f"{search_method}_results"] = rerank_results

    with open(
            f"./user_intent_data/{dataset}/{search_engine}/{search_method}/{rerank_model}-{search_engine}-{dataset}-{split}.jsonl",
            "w",
            encoding="utf-8") as f:
        for data_line in data_lines:
            f.write(json.dumps(data_line, ensure_ascii=False) + "\n")
