import argparse
import concurrent
import os
import json
import re

import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from tqdm import tqdm

# openai_api = os.environ.get('OPENAI_API_ADDR') if os.environ.get('OPENAI_API_ADDR') else "https://api.openai.com"
openai_api = 'http://49.51.186.136'
print("OPENAI_API_ADDR:" + openai_api)
WIKI_NUM_PASSAGES = 20
BING_NUM_SNIPPETS = 20

gpt_rag_prompt = "<<SYS>>\nNow, based on the following reference and your knowledge, please answer the question more succinctly and professionally. The reference is delimited by triple brackets [[[]]]. The question is delimited by triple parentheses ((())). You should include as many possible answers as you can. \n<</SYS>>\nReference: [[[{context}]]], \nquestion: ((({question})))"
gpt_rag_prompt_fluency = "<<SYS>>\nNow, based on the following reference and your knowledge, please answer the question more succinctly and professionally. The reference is delimited by triple brackets [[[]]]. The question is delimited by triple parentheses ((())). You are not allowed to add fabrications or hallucinations. \n<</SYS>>\nReference: [[[{context}]]], \nquestion: ((({question})))"


def chat_prompt_vanilla_search(data_line, search_engine, chat_model):
    if search_engine == "bing":
        _context = [item["snippet"] for item in data_line["vanilla_search_results"][:BING_NUM_SNIPPETS]]
    elif search_engine == "kiltbm25":
        _context = data_line["vanilla_search_results"][:WIKI_NUM_PASSAGES]
    else:
        raise NotImplementedError(f"search engine {search_engine} not implemented")
    context = "\n".join(_context)
    question = data_line["question"]
    prompt = gpt_rag_prompt.format(context=context, question=question)
    while len(tokenizer.encode(prompt)) > 3000:
        #  remove the last 2 sentences
        _context = _context[:-2]
        context = "\n".join(_context)
        prompt = gpt_rag_prompt.format(context=context, question=question)
    return prompt


def chat_prompt_search(data_line, rewrite_model, search_engine, chat_model, use_webcontent=False):
    #  put top 1 search results into context, then top 2, then top 3, then top 4, then top 5
    if search_engine == "bing":
        #  switch column and row
        context = [[item for item in searchword_results[:BING_NUM_SNIPPETS]] for searchword_results in
                   data_line[f"{rewrite_model}_rewrite_search_results"]]
        context = list(map(list, zip(*context)))
        #  flatten list
        context = [item for sublist in context for item in sublist]
        _context = []
        _context_url = set()
        #  get references with different ids
        for i in range(len(context)):
            if len(_context_url) == BING_NUM_SNIPPETS:
                break
            if context[i]["url"] not in _context_url:
                if use_webcontent:
                    _context.append(context[i]["webcontent"])
                else:
                    _context.append(context[i]["snippet"])
                _context_url.add(context[i]["url"])
    elif search_engine == "kiltbm25":
        # take top WIKI_NUM_PASSAGES passages from each search result
        _context = data_line[f"{rewrite_model}_rewrite_search_results"][:WIKI_NUM_PASSAGES]
    else:
        raise NotImplementedError(f"search engine {search_engine} not implemented")

    if len(_context) > 0:
        context = "\n".join(_context)
        question = data_line["question"]
        prompt = gpt_rag_prompt.format(context=context, question=question)
        while len(tokenizer.encode(prompt)) > 3000:
            #  remove the last 2 sentences
            _context = _context[:-2]
            context = "\n".join(_context)
            prompt = gpt_rag_prompt.format(context=context, question=question)
    else:
        question = data_line["question"]
        prompt = f"{question}"
    return prompt


def sending_inference_request(model, model_input):
    if model == "gpt35instruct":
        model = "gpt-3.5-turbo-instruct"
        chat_mode=False
    elif model == "gpt35turbo":
        model = "gpt-3.5-turbo"
        chat_mode=True
    else:
        raise NotImplementedError(f"model {model} not implemented")
    res = proxy.call(model, model_input, chat_mode)
    if chat_mode:
        return res['choices'][0]['message']['content']
    else:
        return res['choices'][0]['text']


class OpenAIApiException(Exception):
    def __init__(self, msg, error_code):
        self.msg = msg
        self.error_code = error_code


class OpenAIApiProxy():
    def __init__(self, api_key=None):
        retry_strategy = Retry(
            total=8,  # 最大重试次数（包括首次请求）
            backoff_factor=10,  # 重试之间的等待时间因子
            status_forcelist=[429, 500, 502, 503, 504],  # 需要重试的状态码列表
            allowed_methods=["POST"]  # 只对POST请求进行重试
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        # 创建会话并添加重试逻辑
        self.session = requests.Session()
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
        self.api_key = api_key

    def call(self, model_name, prompt, chat_mode=False, headers={}):
        if chat_mode:
            params_gpt = {
                "model": model_name,
                "messages": [{"role": "user", "content":prompt}],
                "max_tokens": 1024,
                "temperature": 0.01,
                "top_p": 1,
            }
        else:
            params_gpt = {
                "model": model_name,
                "prompt": prompt,
                "max_tokens": 1024,
                "temperature": 0.01,
                "top_p": 1,
            }
        headers['Content-Type'] = headers['Content-Type'] if 'Content-Type' in headers else 'application/json'
        if self.api_key:
            headers['Authorization'] = "Bearer " + self.api_key
        if chat_mode:
            url = openai_api + '/v1/chat/completions'
        else:
            url = openai_api + '/v1/completions'
        # print(url)
        # print(json.dumps(params_gpt, indent=4, ensure_ascii=False))
        response = self.session.post(url, headers=headers, data=json.dumps(params_gpt))
        if response.status_code != 200:
            err_msg = "access openai error, status code: %s，errmsg: %s" % (response.status_code, response.text)
            raise OpenAIApiException(err_msg, response.status_code)
        data = json.loads(response.text)
        return data


proxy = OpenAIApiProxy()

if __name__ == '__main__':
    #  init 4 threads
    #  each thread has a session
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--dataset", type=str, default="webglm-qa")
    arg_parser.add_argument("--split", type=str, default="test")
    arg_parser.add_argument("--prompt_method", type=str, default="vanilla_search")
    arg_parser.add_argument("--mini_dataset", action="store_true", help="whether to use mini dataset")
    arg_parser.add_argument("--chat_model", type=str, default="gpt35instruct")
    arg_parser.add_argument("--search_engine", type=str, default="kiltbm25")
    arg_parser.add_argument("--rerank_model", type=str, default="e5base")
    args = arg_parser.parse_args()
    dataset = args.dataset
    split = args.split
    prompt_method = args.prompt_method
    chat_model = args.chat_model
    search_engine = args.search_engine
    rerank_model = args.rerank_model

    custom_rewrite_methods = ["llama_rewrite_search", "rule_rewrite_search"]

    if "without_search" in prompt_method:
        if not os.path.exists(f"./user_intent_data/{dataset}/{chat_model}/{prompt_method}/"):
            os.makedirs(f"./user_intent_data/{dataset}/{chat_model}/{prompt_method}/", exist_ok=True)
        input_file = f"./user_intent_data/{dataset}/{dataset}-{split}.jsonl"
        output_file = f"./user_intent_data/{dataset}/{chat_model}/{prompt_method}/{chat_model}-{dataset}-{split}.jsonl"
    else:
        if not os.path.exists(f"./user_intent_data/{dataset}/{chat_model}/{search_engine}/{prompt_method}/"):
            os.makedirs(f"./user_intent_data/{dataset}/{chat_model}/{search_engine}/{prompt_method}/", exist_ok=True)
        if "bing" in search_engine:
            input_file = f"./user_intent_data/{dataset}/{search_engine}/{prompt_method}/{search_engine}-{dataset}-{split}.jsonl"
            output_file = f"./user_intent_data/{dataset}/{chat_model}/{search_engine}/{prompt_method}/{chat_model}-{dataset}-{split}.jsonl"
        else:
            input_file = f"./user_intent_data/{dataset}/{search_engine}/{prompt_method}/{rerank_model}-{search_engine}-{dataset}-{split}.jsonl"
            output_file = f"./user_intent_data/{dataset}/{chat_model}/{search_engine}/{prompt_method}/{rerank_model}-{chat_model}-{dataset}-{split}.jsonl"

    with open(output_file, "w", encoding="utf-8") as f:
        pass
    with open(input_file, "r", encoding="utf-8") as f:
        data_lines = [json.loads(line) for line in f]

    #  tokenize input for llama to truncate inputs which are too long
    tokenizer = None
    if "gpt" in chat_model:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("../../huggingface/llama-2-13b-chat-hf/")
        print(f"loaded tokenizer {tokenizer}")

    if args.mini_dataset:
        data_lines = data_lines[:20]
    #  trim dataset that is too large
    data_lines = data_lines[:5000]
    print(
        f"dataset: {dataset}, split: {split}, prompt_method: {prompt_method}, chat_model: {chat_model}, search_engine: {search_engine}, rerank_model: {rerank_model}")
    # gpt_rag_prompt = gpt_rag_prompt_fluency
    if dataset in ["webglm-qa", "dolly", "eli5"]:
        gpt_rag_prompt = gpt_rag_prompt_fluency
        print(f"using long answer dataset {dataset}, change to fluency prompt for rag.")
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        for idx, data_line in tqdm(enumerate(data_lines), total=len(data_lines),
                                   desc=f"dataset: {dataset}, split: {split}"):
            #  do search with vanilla query
            if "vanilla_search" in prompt_method:
                model_input = chat_prompt_vanilla_search(data_line, search_engine, chat_model)
            elif "gpt4_rewrite_search" in prompt_method:
                use_webcontent = "webcontent" in prompt_method
                model_input = chat_prompt_search(data_line, "gpt4", search_engine, chat_model,
                                                 use_webcontent=use_webcontent)

            elif prompt_method == "without_search":
                model_input = data_line["question"]
            elif re.match(r"v\d[0-9a-z]+_rewrite_search",
                          prompt_method) or prompt_method in custom_rewrite_methods or "plus" in prompt_method:
                model_input = chat_prompt_search(data_line, prompt_method.replace("_rewrite_search", ""), search_engine,
                                                 chat_model)

            else:
                raise NotImplementedError(f"prompt method {prompt_method} not implemented")
            if idx == 1:
                print(model_input)
            future_res = executor.submit(sending_inference_request, chat_model, model_input)
            res = future_res.result()
            data_lines[idx][f"{chat_model}_{prompt_method}_answer"] = res

    with open(output_file, "a", encoding="utf-8") as f:
        #  write to file
        for data_line in data_lines:
            f.write(json.dumps(data_line, ensure_ascii=False) + "\n")
