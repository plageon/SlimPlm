import argparse
import os
import json
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from tqdm import tqdm

# openai_api = os.environ.get('OPENAI_API_ADDR') if os.environ.get('OPENAI_API_ADDR') else "https://api.openai.com"
openai_api = 'http://47.236.144.103'
print("OPENAI_API_ADDR:" + openai_api)


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


# %%
proxy = OpenAIApiProxy("sk-1fdYEsG9Q8nOCdjcB478D8C1Ff6c453b9b266dA5F41f6215")
demo = [{
    "input": "<C_Q>When did Virgin Australia start operating?<NOW>2023-11-20 9:47:28<Pred>",
    "task": "knowledge questions",
    "timeliness": False,
    "entities": ["Virgin Australia"],
    "events": ["Virgin Australia start operating"],
    "questions": [{
        "question": "When did Virgin Australia start operating?",
        "needSearch": True,
        "searchWord": "Virgin Australia start operating time",
    }]
},
    {
        "input": "<C_Q>Give me the top 5 golf equipment company names.<NOW>2023-11-20 9:47:28<Pred>",
        "task": "brain_storming",
        "timeliness": False,
        "entities": [],
        "events": [],
        "questions": [{
            "question": "List top 5 golf equipment company names",
            "needSearch": True,
            "searchWord": "famous golf equipment company names",
        }]

    },
    {
        "input": "<C_Q>What is Israel-Hamas war death toll today for both sides?<NOW>2023-11-20 9:47:28<Pred>",
        "task": "open_qa",
        "timeliness": False,
        "entities": [],
        "events": ["Israel-Hamas war in 2023"],
        "questions": [{
            "question": "What is palestinian death toll today?",
            "needSearch": True,
            "searchWord": "palestinian death toll today",
        },
            {
                "question": "What is israeli death toll today?",
                "needSearch": True,
                "searchWord": "israeli death toll today",
            }]
    },
    {
        "input": "<C_Q>There were 3 birds on the tree. 8 were shot and killed. How many are left?<NOW>2023-11-20 9:47:28<Pred>",
        "task": "mathematical problems",
        "timeliness": False,
        "entities": [],
        "events": [],
        "questions": []
    },
    {
        "input": "<C_Q>Are there any concerts in Beijing recently?<NOW>2023-11-20 9:47:28<Pred>",
        "task": "open_qa",
        "timeliness": True,
        "entities": [],
        "events": [],
        "questions": [{
            "question": "Are there any concerts in Beijing around 11.20?",
            "needSearch": True,
            "searchWord": "concerts in Beijing around 11.20",
        }]
    }
]

prefix_prompt = """
Your task is to perform text analysis on user conversations, and complete the last json item. You need to follow the following rules:

1. Classify user conversations into the following categories: text rewriting, mathematical problems, knowledge questions, text creation, table processing, translation, summarization, logical reasoning, open qa, coding, text classification, information extraction, brainstorming, exams, role-playing, others. The format should be a string and stored in the task field.
2. Determine whether the answer of user input is closely related to current datetime, and store it in the timeliness field in boolean format.
3. If the user’s request involves reasoning, each reasoning process should be described as questions and split into as many sub-questions as possible.
4. The sub-questions after splitting should be placed in the question field in questions, and the sub-questions should be fully described without using pronouns such as “he”, “this”, or “that”.
5. If the sub-question involves very strict factual information such as personal relationships, time, location, policies, regulations, etc., which requires the use of a search engine to answer, then it needs to be marked as needSearch=true, and the generated search term should be placed in searchWord.
6. If the sub-question is a chit-chat question such as "how are you" or a pure mathematical problem, coding, logical reasoning, creative thinking, or common sense problem, then no search is needed.
7. Extract the entities and events involved in the user's request and store them in the entities and events fields respectively. The format is a list of strings. Note that the entities and events should be higly informative, and should not be a user instruction or a question.
"""
model = 'gpt-4-1106-preview'
user_input_prefix = """{
    "input":"<C_Q>"""
user_input_suffix = """<NOW>2023-11-20 9:47:28<Pred>","""

if __name__ == '__main__':
    # dolly_data_file = "./user_intent_data/dolly/databricks-dolly-15k.jsonl"
    # gpt4_data_file = "./user_intent_data/dolly/gpt4/gpt4-databricks-dolly-15k.jsonl"
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--dataset", type=str, default="webglm-qa", help="dataset name")
    arg_parser.add_argument("--split", type=str, default="test", help="split name")
    args = arg_parser.parse_args()
    dataset = args.dataset
    split = args.split

    data_file = f"./user_intent_data/{dataset}/{dataset}-{split}.jsonl"
    gpt4_data_file = f"./user_intent_data/{dataset}/gpt4/unparsed-gpt4-{dataset}-{split}.jsonl"
    data_lines = [json.loads(line) for line in open(data_file, "r", encoding="utf-8")]
    if len(data_lines) > 15000:
        data_lines = data_lines[:15000]
    if not os.path.exists(gpt4_data_file):
        os.makedirs(os.path.dirname(gpt4_data_file), exist_ok=True)
        with open(gpt4_data_file, "w", encoding="utf-8") as f:
            pass
    gpt4_resutls = data_lines.copy()
    for idx, data_line in tqdm(enumerate(data_lines), total=len(data_lines), desc="gpt4"):
        question = data_line["question"] if "question" in data_line else data_line["instruction"]
        prompt = prefix_prompt + json.dumps(demo, indent=4, ensure_ascii=False)[:-2] + user_input_prefix + \
                 question + user_input_suffix
        res = proxy.call(model, prompt)
        # print(res['choices'][0]['message']['content'])
        gpt4_resutls[idx]["preds"] = res['choices'][0]['message']['content']

        with open(gpt4_data_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(gpt4_resutls[idx], ensure_ascii=False) + "\n")
