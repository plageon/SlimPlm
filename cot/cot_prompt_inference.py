import argparse
import concurrent
import json
import os

import requests
import torch
from tqdm import tqdm

params_baichuan = {"repetition_penalty": 1.05, "temperature": 0.01, "top_k": 1, "top_p": 0.85, "max_new_tokens": 512,
                   "do_sample": False, "seed": 2023}
headers = {'Content-Type': 'application/json'}

def sending_inference_request(question):
    if "qwen" in chat_model and local_model is not None:
        response, history = local_model.chat(query=question, history=None)
        # response, history = local_model.chat(tokenizer, question, history=None)
        return response
    else:
        data = {"inputs": question, "parameters": params_baichuan}
        request = requests.post(INFERENCE_URL, json=data, headers=headers)

        return json.loads(request.text)[0]['generated_text']

cot_prompts = {
    "asqa": "Question: {question}\nLet’s think step by step.",
    "nq": "Question: {question}\nLet’s think step by step.",
    "trivia-qa": "Question: {question}\nLet’s think step by step.",
    "eli5": "Question: {question}\nLet’s think step by step.",
    "hotpot-qa": """Question: What is the name of this American musician, singer, actor, comedian, and songwriter, who worked with Modern Records and born in December 5, 1932?
Let’s think step by step.
Artists who worked with Modern Records include Etta James, Joe Houston, Little Richard, Ike and Tina Turner and John Lee Hooker in the 1950s and 1960s. Of these Little Richard, born in December 5, 1932, was an American musician, singer, actor, comedian, and songwriter.
So the answer is Little Richard

Question: Between Chinua Achebe and Rachel Carson, who had more diverse jobs?
Let’s think step by step.
Chinua Achebe was a Nigerian novelist, poet, professor, and critic. Rachel Carson was an American marine biologist, author, and conservationist. So Chinua Achebe had 4 jobs, while Rachel Carson had 3 jobs. Chinua Achebe had more diverse jobs than Rachel Carson.
So the answer is Chinua Achebe

Question: Remember Me Ballin’ is a CD single by Indo G that features an American rapper born in what year?
Let’s think step by step.
Remember Me Ballin’ is the CD single by Indo G featuring Gangsta Boo. Gangsta Boo is Lola Mitchell’s stage name, who was born in August 7, 1979, and is an American rapper.
So the answer is 1979
Question: {question}
Let’s think step by step.""",
    "2wiki": """Question: Which film came out first, Blind Shaft or The Mask Of Fu Manchu?
Let’s think step by step.
Blind Shaft is a 2003 film, while The Mask Of Fu Manchu opened in New York on December 2, 1932. 2003 comes after 1932. Therefore, The Mask Of Fu Manchu came out earlier than Blind Shaft. So the answer is The Mask Of Fu Manchu

Question: When did John V, Prince Of Anhalt-Zerbst’s father die?
Let’s think step by step.
John was the second son of Ernest I, Prince of Anhalt-Dessau. Ernest I, Prince of Anhalt-Dessau died on 12 June 1516.
So the answer is 12 June 1516

Question: Which film has the director who was born later, El Extrano Viaje or Love In Pawn?
Let’s think step by step.
The director of El Extrano Viaje is Fernando Fernan Gomez, who was born on 28 August 1921. The director of Love In Pawn is Charles Saunders, who was born on 8 April 1904. 28 August 1921 comes after 8 April 1904. Therefore, Fernando Fernan Gomez was born later than Charles Saunders.
So the answer is El Extrano Viaje
Question: {question}
Let’s think step by step.""",
    "musique": """Question: In which year did the publisher of In Cold Blood form?
Let’s think step by step.
In Cold Blood was first published in book form by Random House. Random House was form in 2001.
So the answer is 2001

Question: Who was in charge of the city where The Killing of a Sacred Deer was filmed?
Let’s think step by step.
The Killing of a Sacred Deer was filmed in Cincinnati. The present Mayor of Cincinnati is John Cranley. Therefore, John Cranley is in charge of the city.
So the answer is John Cranley

Question: Where on the Avalon Peninsula is the city that Signal Hill overlooks?
Let’s think step by step.
Signal Hill is a hill which overlooks the city of St. John’s. St. John’s is located on the eastern tip of the Avalon Peninsula.
So the answer is eastern tip
Question: {question}
Let’s think step by step."""
}

if __name__ == '__main__':
    #  init 4 threads
    #  each thread has a session
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--dataset", type=str, default="webglm-qa")
    arg_parser.add_argument("--split", type=str, default="test")
    arg_parser.add_argument("--prompt_method", type=str, default="vanilla_search")
    arg_parser.add_argument("--mini_dataset", action="store_true", help="whether to use mini dataset")
    arg_parser.add_argument("--chat_model", type=str, default="llama13b")
    arg_parser.add_argument("--inference_url", type=str, default="")
    args = arg_parser.parse_args()
    dataset = args.dataset
    split = args.split
    prompt_method = args.prompt_method
    chat_model = args.chat_model
    INFERENCE_URL = args.inference_url
    assert "cot" in prompt_method, f"prompt_method {prompt_method} is not a cot prompt method"

    #  tokenize input for llama to truncate inputs which are too long
    tokenizer = None
    local_model = None
    if "llama" in chat_model:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("../../huggingface/llama-2-13b-chat-hf/")
        print(f"loaded tokenizer {tokenizer}")
    if "qwen" in chat_model:
        from transformers import AutoTokenizer, AutoModelForCausalLM

        tokenizer = AutoTokenizer.from_pretrained("../../huggingface/Qwen-7B-Chat/", trust_remote_code=True)
        print(f"loaded tokenizer {tokenizer}")
        #  load model locally
        qwen_model_names = {
            "qwen7b": "Qwen-7B-Chat",
            "qwen14b": "Qwen-14B-Chat",
            "qwen72b": "Qwen-72B-Chat",
        }
        from vllm_wrapper import vLLMWrapper

        gpu_device_count = torch.cuda.device_count()
        local_model = vLLMWrapper(f'../../huggingface/{qwen_model_names[chat_model]}/',
                                  tensor_parallel_size=gpu_device_count)
        # local_model = AutoModelForCausalLM.from_pretrained(f"../../huggingface/{qwen_model_names[chat_model]}/", device_map="auto", trust_remote_code=True)
        # if torch.cuda.is_available():
        #     local_model = local_model.cuda()
        # local_model.eval()
        print(f"loaded model {qwen_model_names[chat_model]}")

    if not os.path.exists(f"./user_intent_data/{dataset}/{chat_model}/{prompt_method}/"):
        os.makedirs(f"./user_intent_data/{dataset}/{chat_model}/{prompt_method}/", exist_ok=True)
    input_file = f"./user_intent_data/{dataset}/{dataset}-{split}.jsonl"
    output_file = f"./user_intent_data/{dataset}/{chat_model}/{prompt_method}/{chat_model}-{dataset}-{split}.jsonl"

    with open(output_file, "w", encoding="utf-8") as f:
        pass
    with open(input_file, "r", encoding="utf-8") as f:
        data_lines = [json.loads(line) for line in f]

    if args.mini_dataset:
        data_lines = data_lines[:20]
    #  trim dataset that is too large
    data_lines = data_lines[:5000]
    print(
        f"dataset: {dataset}, split: {split}, prompt_method: {prompt_method}, chat_model: {chat_model}")
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        for idx, data_line in tqdm(enumerate(data_lines), total=len(data_lines),
                                   desc=f"dataset: {dataset}, split: {split}"):
            question = data_line["question"]
            prompt= cot_prompts[dataset].format(question=question)
            if "baichuan" in chat_model:
                model_input = "<C_Q> " + prompt + " <C_A>"
            elif "llama" in chat_model:
                model_input = "<s>[INST] " + prompt + " [/INST]"
            elif "qwen" in chat_model:
                model_input = prompt
            else:
                raise NotImplementedError(f"chat model {chat_model} not implemented")

            if idx == 1:
                print(model_input)
            future_res = executor.submit(sending_inference_request, model_input)
            res = future_res.result()
            data_lines[idx][f"{chat_model}_{prompt_method}_answer"] = res

    with open(output_file, "a", encoding="utf-8") as f:
        #  write to file
        for data_line in data_lines:
            f.write(json.dumps(data_line, ensure_ascii=False) + "\n")

