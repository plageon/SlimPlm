import argparse
import json
import os
import re
from time import sleep

import requests
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM, LlamaTokenizer, AutoModel
import torch.nn.functional as F
from torch import Tensor

llama_flare_prompt = "<s>[INST]<<SYS>>\nYou are a helpful assistant. \n<<SYS>>\n {question} \n[/INST]{history}"
qwen_flare_prompt = "<|im_start|>You are a helpful assistant.<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n{history}"

from transformers import AutoTokenizer, AutoConfig, AutoModelForSeq2SeqLM
from transformers.integrations import HfDeepSpeedConfig
import deepspeed
import os
import torch


# local_rank = int(os.getenv("LOCAL_RANK", "0"))
# world_size = int(os.getenv("WORLD_SIZE", "1"))
#

def init_ds_config(model_name):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # To avoid warnings about parallelism in tokenizers

    # distributed setup

    deepspeed.init_distributed()

    config = AutoConfig.from_pretrained(model_name)
    model_hidden_size = config.d_model

    # batch size has to be divisible by world_size, but can be bigger than world_size
    ds_config = {
        "mp_size": 4,  # model parallel size
        # "tp_size": 4,  # tensor parallel size
        "fp16": {
            "enabled": False
        },
        "bf16": {
            "enabled": True
        },
        "zero_optimization": {
            "stage": 3,
            "offload_param": {
                "device": "cpu",
                "pin_memory": True
            },
            # "overlap_comm": True,
            # "contiguous_gradients": True,
            # "reduce_bucket_size": model_hidden_size * model_hidden_size,
            # "stage3_prefetch_bucket_size": 0.9 * model_hidden_size * model_hidden_size,
            # "stage3_param_persistence_threshold": 10 * model_hidden_size
        },
        "steps_per_print": 2000,
        "wall_clock_breakdown": False
    }
    return ds_config


def cal_logits(question, history, answer):
    if chat_model == "qwen72b":
        model_input = qwen_flare_prompt.format(question=question, history=history)
    elif chat_model == "llama70b":
        model_input = llama_flare_prompt.format(question=question, history=history)
    else:
        raise NotImplementedError(f"chat model {chat_model} not implemented")
    input_ids = tokenizer.encode(model_input, add_special_tokens=False)
    target_ids = tokenizer.encode(answer, add_special_tokens=False)
    input_tensor = torch.tensor([input_ids + target_ids]).to(model.device)
    target_tensor = torch.tensor([[-100] * len(input_ids) + target_ids]).to(model.device)
    with torch.no_grad():
        outputs = model(input_tensor, labels=target_tensor)

        # loss is calculated using CrossEntropyLoss which averages over valid labe
        # N.B. the rerank_model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
        # to the left by 1.
        logits = outputs.logits

        logits = logits[0, len(input_ids):, :]
        logits = logits.to(torch.float32).detach().cpu().numpy()
        #  get the logits of the target ids
        logits = logits[range(len(target_ids)), target_ids]
    return target_ids, logits


def check_chat_result(question, answer, pre_history):
    # assert len(answer) >= len(
    #     pre_history) and pre_history in answer, f"pre_history \n{pre_history} \nis not in answer \n{answer}"
    # if len(answer) == len(pre_history):
    #     return "", ""
    #  last sentence position in the pre_history, including ., !, ?
    if pre_history:
        pre_history = pre_history.strip()
        #  move the first sentence from answer to pre_history
        answer_first_sentence = [answer.find(s) for s in [".", "?", "!"] if s in answer]
        if len(answer_first_sentence) > 0:
            answer_first_sentence = min(answer_first_sentence)
            pre_history += answer[:answer_first_sentence + 1]
            answer = answer[answer_first_sentence + 1:]
        else:
            pre_history += answer
            answer = ""
    answer = answer.strip()
    if len(answer) == 0:
        return "", ""

    target_ids, logits = cal_logits(question, pre_history, answer)
    #  segment target ids and logits by sentence end token
    target_ids_list = []
    logits_list = []
    if "llama" in chat_model:
        #  segment text sentence first
        # split sentence by . ? !
        text_sentences = re.split(r'(?<=[^A-Z].[.?]) +', answer)
        #  encode the sentences and the entire text
        token_id_sentences = [tokenizer.encode(s, add_special_tokens=False) for s in text_sentences]
        assert sum([len(s) for s in token_id_sentences]) == len(
            target_ids), f"token id sentences length {sum([len(s) for s in token_id_sentences])} not equal to target ids length {len(target_ids)}\n{token_id_sentences}\n{text_sentences}\n{answer}"
        target_ids_list = token_id_sentences
        logits_list = [logits[:len(s)] for s in token_id_sentences]

    elif "qwen" in chat_model:
        #  segment text sentence first
        # split sentence by . ? !
        start = 0
        for i, t in enumerate(target_ids):
            if t in senentce_end_token_ids:
                target_ids_list.append(target_ids[start:i + 1])
                logits_list.append(logits[start:i + 1])
                start = i + 1

        #  in case the last token is not sentence end token
        if start < len(target_ids):
            target_ids_list.append(target_ids[start:])
            logits_list.append(logits[start:])
    else:
        raise NotImplementedError(f"chat model {chat_model} not implemented")

    #  check the first sentence that contains logits smaller than threshold
    #  the current sentence is the last sentence
    #  the sentence before the current sentence is history
    last_sentence = None
    for i, snt_logits in enumerate(logits_list):
        for l in snt_logits:
            if l < threshold:
                last_sentence = i
                break

    #  mask the low prob token to reformat the last sentence to query

    if last_sentence is not None:
        #  remove low prob token
        last_sentence_target_ids = target_ids_list[last_sentence]
        last_sentence_logits = logits_list[last_sentence]
        mask = [0 if l < threshold else 1 for l in last_sentence_logits]
        query_ids = [t for t, m in zip(last_sentence_target_ids, mask) if m]

        #  the sentence before the last sentence and the content before the first low prob token in the last sentence is history
        history = []
        for i in range(last_sentence):
            history.extend(target_ids_list[i])
        history.extend(last_sentence_target_ids[:mask.index(0)])
        history = pre_history + tokenizer.decode(history)
        #  the last sentence is the new query
        query = tokenizer.decode(query_ids)
        # print(f"history: {history}")
        # print(f"query: {query}")
        return history, query
    else:
        return "", ""


def apply_retrieval(question):
    #  the question is a string
    if search_engine == "kiltbm25":
        patience = 3
        search_results = []
        while True:
            try:
                body = {'queries': [question], 'question': question}
                if search_engine == "kiltbm25":
                    res = requests.request('GET', 'http://172.17.31.12:5050/search/bm25_e5_rerank/', json=body)
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


def first_round(data_lines):
    # calculate the logits
    for idx, chat in enumerate(tqdm(data_lines)):
        query = chat["question"]
        answer = chat[f"{chat_model}_without_search_answer"]
        #  check the chat result
        history, query = check_chat_result(query, answer, "")
        chat["history"] = [history]
        chat["query"] = [query]
        #  if query is empty, skip retrieval
        if query == "":
            chat["search_results"] = [""]
        else:
            #  do retrieval

            search_results = apply_retrieval(query)
            chat["search_results"] = [search_results]
    return data_lines


def later_round(data_lines):
    # calculate the logits
    for idx, chat in enumerate(tqdm(data_lines)):
        #  if former round is empty, skip retrieval
        if chat["query"][round - 1] == "":
            chat["search_results"].append("")
            chat["history"].append("")
            chat["query"].append("")
            continue
        #  check the chat result
        assert len(chat["query"]) == round
        latest_history = chat["history"][-1]
        history, query = check_chat_result(chat["question"], chat[f"{chat_model}_round_answer"][-1], latest_history)
        chat["history"].append(history)
        chat["query"].append(query)
        #  if query is empty, skip retrieval
        if query == "":
            chat["search_results"].append("")
        else:
            #  do retrieval
            search_results = apply_retrieval(query)
            chat["search_results"].append(search_results)
    return data_lines


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--dataset", type=str, default="webglm-qa")
    arg_parser.add_argument("--split", type=str, default="test")
    arg_parser.add_argument("--chat_model", type=str, default="llama13b")
    arg_parser.add_argument("--search_engine", type=str, default="kiltbm25")
    arg_parser.add_argument("--round", type=int, default=0)
    arg_parser.add_argument("--mini_dataset", action="store_true")
    args = arg_parser.parse_args()
    dataset = args.dataset
    split = args.split
    chat_model = args.chat_model
    search_engine = args.search_engine
    round = args.round

    threshold = 0.5
    model_path = {
        "llama70b": "../../huggingface/llama-2-70b-chat-hf/",
        "qwen72b": "../../huggingface/Qwen-72B-Chat/",
    }
    tokenizer = AutoTokenizer.from_pretrained(model_path[chat_model], trust_remote_code=True)
    print(f"loaded tokenizer {tokenizer}")
    #  get the token id of . ? !
    senentce_end_token_ids = [tokenizer.encode(s, add_special_tokens=False)[0] for s in [".", "?", "!"]]

    # Initialize the model in 8 bit mode

    # max_memory_mapping = {0: "79GB", 1: "79GB", 2: "79GB", 3: "79GB"}
    max_memory_mapping = {0: "79GB", 1: "79GB"}
    model = AutoModelForCausalLM.from_pretrained(
        model_path[chat_model], device_map="auto", load_in_8bit=True, max_memory=max_memory_mapping,
        trust_remote_code=True
    ).eval()

    print(f"loaded model {model}")
    print(
        f"dataset: {dataset}, split: {split}, chat_model: {chat_model}, search_engine: {search_engine}, round: {round}")
    if round == 0:
        #  open without retrieval result
        no_search_chat_file = f"./user_intent_data/{dataset}/{chat_model}/without_search/{chat_model}-{dataset}-{split}.jsonl"
        data_lines = [json.loads(line) for line in open(no_search_chat_file, "r", encoding="utf-8")]
    else:
        input_file = f"./user_intent_data/{dataset}/{chat_model}/flare/{chat_model}round{round}chat-{dataset}-{split}.jsonl"
        data_lines = [json.loads(line) for line in open(input_file, "r", encoding="utf-8")]
    if args.mini_dataset:
        data_lines = data_lines[:20]

    output_file = f"./user_intent_data/{dataset}/{chat_model}/flare/{chat_model}round{round}-{dataset}-{split}.jsonl"


    if round == 0:
        if not os.path.exists(f"./user_intent_data/{dataset}/{chat_model}/flare/"):
            os.makedirs(f"./user_intent_data/{dataset}/{chat_model}/flare/", exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            pass
        searched_data_lines = first_round(data_lines)
        #  save the result
        with open(output_file, "a", encoding="utf-8") as f:
            for chat in searched_data_lines:
                f.write(json.dumps(chat) + "\n")
    else:
        searched_data_lines = later_round(data_lines)
        with open(output_file, "w", encoding="utf-8") as f:
            pass
        #  save the result
        with open(output_file, "a", encoding="utf-8") as f:
            for chat in searched_data_lines:
                f.write(json.dumps(chat) + "\n")
