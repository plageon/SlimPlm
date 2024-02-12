import re
import concurrent.futures

from tqdm import tqdm
import argparse
import os
import json
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from tqdm import tqdm

# openai_api = os.environ.get('OPENAI_API_ADDR') if os.environ.get('OPENAI_API_ADDR') else "https://api.openai.com"
openai_api = 'http://47.236.144.103'


class OpenAIApiException(Exception):
    def __init__(self, msg, error_code):
        self.msg = msg
        self.error_code = error_code


class OpenAIApiProxy():
    def __init__(self, api_key=None):
        retry_strategy = Retry(
            total=8,  # 最大重试次数（包括首次请求）
            backoff_factor=10,  # 重试之间的等待时间因子
            status_forcelist=[429, 500, 502, 503, 504, 404],  # 需要重试的状态码列表
            allowed_methods=["POST"]  # 只对POST请求进行重试
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        # 创建会话并添加重试逻辑
        self.session = requests.Session()
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
        self.api_key = api_key

    def call(self, model_name, prompt, headers={}):
        params_gpt = {
            "model": model_name,
            "messages": [{"role": "user", "content": ''}],
            "max_tokens": 512,
            "temperature": 0.01,
            "top_p": 1,
        }
        params_gpt['model'] = model_name
        params_gpt['messages'][0]['content'] = prompt
        headers['Content-Type'] = headers['Content-Type'] if 'Content-Type' in headers else 'application/json'
        if self.api_key:
            headers['Authorization'] = "Bearer " + self.api_key
        url = openai_api + '/v1/chat/completions'
        # print(url)
        # print(json.dumps(params_gpt, indent=4, ensure_ascii=False))
        response = self.session.post(url, headers=headers, data=json.dumps(params_gpt))
        if response.status_code != 200:
            err_msg = "access openai error, status code: %s，errmsg: %s" % (response.status_code, response.text)
            raise OpenAIApiException(err_msg, response.status_code)
        data = json.loads(response.text)
        return data


params_baichuan = {"repetition_penalty": 1.05, "temperature": 0.01, "top_k": 1, "top_p": 0.85, "max_new_tokens": 512,
                   "do_sample": False, "seed": 2023}
headers = {'Content-Type': 'application/json'}
WIKI_NUM_PASSAGES = 20
BING_NUM_SNIPPETS = 20

demo = """In the 2016 presidential election, the state of Colorado was won by the Democratic candidate,  Hillary Clinton. She received 48.9% of the vote, while the Republican candidate, Donald Trump, received 43.7% of the vote. The Libertarian candidate, Gary Johnson, received 4.9% of the vote, and the Green Party candidate, Jill Stein, received 2.5% of the vote. [/INST] <claims> The state of Colorado was won by the Democratic candidate, Hillary Clinton, in the 2016 presidential election.***** Hillary Clinton received 48.9% of the vote in Colorado.***** The Republican candidate, Donald Trump, received 43.7% of the vote in Colorado.***** The Libertarian candidate, Gary Johnson, received 4.9% of the vote in Colorado.***** The Green Party candidate, Jill Stein, received 2.5% of the vote in Colorado. </claims> """

baichuan_sep_prompt = """<C_Q> {question} <C_A>"""
llama_sep_prompt = """<s>[INST] <SYS>>You are asked to separate a given text by claims using separator:'*****'.
Here are some requirements:
1. The separation is conducted according to the meaning and each claim should be be brief and contain as one key claim.
2. Do not add any hallucinated information or miss any information.
3. The claims should be independent and self-contained, and the claims should be fully described without using pronouns such as "he", "this", or "that".
4. The final return should be claims separated with separator:'*****', and embraced by '<claims>' and '</claims>'.
Do strictly in format like this: 
\t<claims> [claim1]*****[claim2]*****[claim3]*****......</claims> <</SYS>>\n {demo} </s><s> [INST] \n{text} [/INST] """

gpt_sep_prompt = ("""<<SYS>>You are asked to first separate a given text by claims and then provide a search query to verify each claim if needed.
Here are some requirements:
1. The separation is conducted according to the meaning and each claim should be be brief and contain as one key claim.
2. Do not add any hallucinated information or miss any information.
3. The claims should be independent and self-contained, and the claims should be fully described without using pronouns such as “he”, “this”, or “that”.
4. The query is derived from it's corresponding claim and the original user question, and should be useful to check the factuality of the claim.
5. If the claim does not contain any fact relevant with the original user question, or only contains simple commen senses, then search is not required.
6. The final return should strictly follow the given format.
Like this: <Claims> <Claim(claim1)> <Search(True/False)> <Query(query1)> <Claim(claim2)> <Search(True/False)> <Query(query2)> <Claim(claim3)><Search(True/False)><Query(query3)>......</Claims> <</SYS>>

[INST] <Question(who won colorado in the 2016 presidential election)> <Text(In the 2016 presidential election, the state of Colorado was won by the Democratic candidate,  Hillary Clinton. She received 48.9% of the vote, while the Republican candidate, Donald Trump, received 43.7% of the vote. The Libertarian candidate, Gary Johnson, received 4.9% of the vote, and the Green Party candidate, Jill Stein, received 2.5% of the vote.)>[/INST] 
<Claims> <Claim(The state of Colorado was won by the Democratic candidate, Hillary Clinton, in the 2016 presidential election)> <Search(True)> <Query(Hillary Clinton 2016 presidential election Colorado result)> 
<Claim<Hillary Clinton received 48.9% of the vote in Colorado)> <Search(True)> <Query(Hillary Clinton Colorado vote percentage 2016)> 
<Claim(The Republican candidate, Donald Trump, received 43.7% of the vote in Colorado)> <Search(True)> <Query(Donald Trump Colorado vote percentage 2016)>
<Claim(The Libertarian candidate, Gary Johnson, received 4.9% of the vote in Colorado)> <Search(True)> <Query(Gary Johnson Colorado vote percentage 2016)> <Claim(The Green Party candidate, Jill Stein, received 2.5% of the vote in Colorado)> <Search(True)> <Query(Jill Stein Colorado vote percentage 2016)>
</Claims>  
[INST] <Question(A sevruga is what type of creature?)> <Text(There is no such thing as a "sevruga." It does not appear to be a real or fictional creature. It may be a misspelling or a made-up word with no meaning or context. If you have any more information or clarification about the term "sevruga," I would be happy to try and assist you further.)> [/INST]
<Claims> 
<Claim(There is no such thing as a "sevruga" as a real or fictional creature)> <Search(True)> <Query(is sevruga a real or fictional creature)>
<Claim("Sevruga" may be a misspelling or a made-up word with no meaning or context)> <Search(True)> <Query(meaning of the word sevruga)>
</Claims> 
[INST] <Question(Where is true grit supposed to take place?)> <Text(True Grit is a novel by Charles Portis that was published in 1968. The story is set in the Arkansas Ozarks during the early 1870s. The novel follows the adventures of Mattie Ross, a young girl who seeks to avenge her father's murder by hiring a U.S. Marshal named Rooster Cogburn to track down and bring to justice the man who killed him.

The story takes place in various locations throughout the Ozarks, including the town of Fort Smith, Arkansas, and the surrounding countryside. The novel is known for its vivid descriptions of the rugged landscape and the colorful characters who inhabit it.

The 2010 film adaptation of True Grit, directed by the Coen brothers, was filmed on location in Arkansas and features many of the same settings as the novel. The film was shot in and around the towns of Bentonville, Fayetteville, and Eureka Springs, among other locations.)> [/INST]
<Claims> 
<Claim(True Grit is a novel by Charles Portis, published in 1968)> <Search(False)> 
<Claim(The story of True Grit is set in the Arkansas Ozarks during the early 1870s)> <Search(True)> <Query(True Grit novel setting)>
<Claim(The novel True Grit follows the adventures of Mattie Ross, a young girl who seeks to avenge her father's murder)> <Search(False)> 
<Claim(True Grit includes the character U.S. Marshal Rooster Cogburn, hired by Mattie Ross to track down her father's murderer)> <Search(False)> 
<Claim(The story of True Grit takes place in various locations throughout the Ozarks, including Fort Smith, Arkansas, and the surrounding countryside)> <Search(True)> <Query(Locations in True Grit novel)>
<Claim(The novel True Grit is known for its vivid descriptions of the rugged landscape and the colorful characters who inhabit it)> <Search(False)> 
<Claim(The 2010 film adaptation of True Grit was directed by the Coen brothers and filmed on location in Arkansas)> <Search(True)> <Query(True Grit 2010 film filming locations)>
<Claim(The 2010 True Grit film was shot in and around the towns of Bentonville, Fayetteville, and Eureka Springs, Arkansas)> <Search(True)> <Query(Filming locations of 2010 True Grit film)>
</Claims>  
[INST] <Question({question})> <Text({text})> [INST]""")


def sending_inference_request(question):
    data = {"inputs": question, "parameters": params_baichuan}
    request = requests.post(INFERENCE_URL, json=data, headers=headers)

    return json.loads(request.text)[0]['generated_text']


if __name__ == '__main__':
    #  init 4 threads
    #  each thread has a session
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--dataset", type=str, default="webglm-qa")
    arg_parser.add_argument("--split", type=str, default="test")
    arg_parser.add_argument("--mini_dataset", action="store_true", help="whether to use mini dataset")
    arg_parser.add_argument("--chat_model", type=str, default="llama13b")
    arg_parser.add_argument("--sep_model", type=str, default="llama13b")
    arg_parser.add_argument("--search_engine", type=str, default="kiltbm25")
    arg_parser.add_argument("--rerank_model", type=str, default="e5base")
    arg_parser.add_argument("--prompt_method", type=str, default="without_search")
    args = arg_parser.parse_args()
    dataset = args.dataset
    split = args.split
    prompt_method = args.prompt_method
    chat_model = args.chat_model
    sep_model = args.sep_model
    search_engine = args.search_engine
    rerank_model = args.rerank_model
    _prompt_method = prompt_method if not prompt_method.endswith("_webcontent") else prompt_method[:-11]
    assert _prompt_method in ["vanilla_search", "gpt4_rewrite_search", "without_search", "llama_rewrite_search",
                              "rule_rewrite_search",
                              "gold_search", "gpt4_rewrite_search"] or re.match(r"v\d+_rewrite_search", _prompt_method)

    custom_rewrite_methods = ["llama_rewrite_search", "rule_rewrite_search"]
    proxy = None
    openai_model = None
    if sep_model == "llama13b":
        INFERENCE_URL = "http://172.17.58.204:80"
    elif sep_model == "baichuan":
        INFERENCE_URL = "http://10.5.0.6:8206"
    elif sep_model == "gpt4":
        openai_model = "gpt-4-1106-preview"
        INFERENCE_URL = "http://49.51.186.136"
        proxy = OpenAIApiProxy("sk-1fdYEsG9Q8nOCdjcB478D8C1Ff6c453b9b266dA5F41f6215")
    elif sep_model == "gpt3":
        openai_model = "gpt-4-1106-preview"
        INFERENCE_URL = "http://49.51.186.136"
        proxy = OpenAIApiProxy()
    else:
        raise NotImplementedError(f"chat model {chat_model} not implemented")
    print(f"INFERENCE_URL: {INFERENCE_URL}")

    if "without_search" in prompt_method:
        if not os.path.exists(f"./user_intent_data/{dataset}/{chat_model}/{prompt_method}/"):
            os.makedirs(f"./user_intent_data/{dataset}/{chat_model}/{prompt_method}/", exist_ok=True)
        input_file = f"./user_intent_data/{dataset}/{chat_model}/{prompt_method}/{chat_model}-{dataset}-{split}.jsonl"
        output_file = f"./user_intent_data/{dataset}/{chat_model}/{prompt_method}/unparsed-{sep_model}sep-{chat_model}-{dataset}-{split}.jsonl"
    else:
        if not os.path.exists(f"./user_intent_data/{dataset}/{chat_model}/{search_engine}/{prompt_method}/"):
            os.makedirs(f"./user_intent_data/{dataset}/{chat_model}/{search_engine}/{prompt_method}/", exist_ok=True)
        if "bing" in search_engine:
            input_file = f"./user_intent_data/{dataset}/{chat_model}/{search_engine}/{prompt_method}/{chat_model}-{dataset}-{split}.jsonl"
            output_file = f"./user_intent_data/{dataset}/{chat_model}/{search_engine}/{prompt_method}/unparsed-{sep_model}sep-{chat_model}-{dataset}-{split}.jsonl"
        else:
            input_file = f"./user_intent_data/{dataset}/{chat_model}/{search_engine}/{prompt_method}/{rerank_model}-{chat_model}-{dataset}-{split}.jsonl"
            output_file = f"./user_intent_data/{dataset}/{chat_model}/{search_engine}/{prompt_method}/unparsed-{sep_model}sep-{rerank_model}-{chat_model}-{dataset}-{split}.jsonl"

    # with open(output_file, "w", encoding="utf-8") as f:
    #     pass
    with open(input_file, "r", encoding="utf-8") as f:
        data_lines = [json.loads(line) for line in f]

    #  tokenize input for llama to truncate inputs which are too long
    tokenizer = None
    if "llama" in sep_model:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("../../huggingface/llama-2-13b-chat-hf/")
        print(f"loaded tokenizer {tokenizer}")

    if args.mini_dataset:
        data_lines = data_lines[:10]
    # data_lines = data_lines[994:]
    print(
        f"dataset: {dataset}, split: {split}, prompt_method: {prompt_method}, chat_model: {chat_model}, search_engine: {search_engine}, rerank_model: {rerank_model}")

    if "gpt" in sep_model:
        for idx, data_line in tqdm(enumerate(data_lines), total=len(data_lines),
                                   desc=f"dataset: {dataset}, split: {split}"):
            answer = data_line[f"{chat_model}_{prompt_method}_answer"]
            model_input = gpt_sep_prompt.format(question=data_line["question"], text=answer)
            data_lines[idx][f"{chat_model}_{prompt_method}_{sep_model}sep_claims"] = proxy.call(openai_model,
                                                                                                model_input)
            with open(output_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(data_line, ensure_ascii=False) + "\n")
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            for idx, data_line in tqdm(enumerate(data_lines), total=len(data_lines),
                                       desc=f"dataset: {dataset}, split: {split}"):
                #  separate answer into claims
                answer = data_line[f"{chat_model}_{prompt_method}_answer"]
                #  remove sentences end with ?
                # sentences = re.split(r'(?<=[.!?]) +', answer)
                # # Filtering out sentences that end with a question mark
                # filtered_sentences = [sentence for sentence in sentences if not sentence.endswith('?')]
                #
                # # Joining the remaining sentences
                # answer = ' '.join(filtered_sentences)

                model_input = llama_sep_prompt.format(text=answer, demo=demo)

                if idx == 1:
                    print(f"model_input: {model_input}")
                future_res = executor.submit(sending_inference_request, model_input)
                res = future_res.result()
                data_lines[idx][f"{chat_model}_{prompt_method}_{sep_model}sep_claims"] = res['choices'][0]['message'][
                    'content']

        with open(output_file, "a", encoding="utf-8") as f:
            #  write to file
            for data_line in data_lines:
                f.write(json.dumps(data_line, ensure_ascii=False) + "\n")
