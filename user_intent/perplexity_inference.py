import argparse
import json
import os

import torch
from tqdm import tqdm
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM

llama_rag_prompt = "<s>[INST] <<SYS>>\nNow, based on the following reference and your knowledge, please answer the question more succinctly and professionally. The reference is delimited by triple brackets [[[]]]. The question is delimited by triple parentheses ((())). You should include as many possible answers as you can. \n<</SYS>>\nReference: [[[{context}]]], \nquestion: ((({question}))) [/INST]"
llama_rag_prompt_fluency = "<s>[INST] <<SYS>>\nNow, based on the following reference and your knowledge, please answer the question more succinctly and professionally. The reference is delimited by triple brackets [[[]]]. The question is delimited by triple parentheses ((())). You are not allowed to add fabrications or hallucinations. \n<</SYS>>\nReference: [[[{context}]]], \nquestion: ((({question}))) [/INST]"

def cal_ppl(query, answer):
    input_ids = tokenizer.encode(query, add_special_tokens=False)
    target_ids = tokenizer.encode(answer, add_special_tokens=False)
    input_tensor = torch.tensor([input_ids + target_ids]).to(model.device)
    target_tensor = torch.tensor([[-100]*len(input_ids)+ target_ids]).to(model.device)
    with torch.no_grad():
        outputs = model(input_tensor,  labels=target_tensor)

        # loss is calculated using CrossEntropyLoss which averages over valid labe
        # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
        # to the left by 1.
        neg_log_likelihood = outputs.loss
        #print(neg_log_likelihood)
        #nlls.append(neg_log_likelihood)
        ppl = neg_log_likelihood.to(torch.float32).detach().cpu().item()
        #ppl = torch.exp(torch.stack(nlls).mean()).to(torch.float32).detach().cpu().numpy()
    return ppl

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

    if chat_model== "llama13b":
        model_path = "../../huggingface/llama-2-70b-chat-hf"
    elif chat_model == "llama70b":
        model_path = "../../huggingface/llama-2-70b-chat-hf"
    elif chat_model == "llama7b":
        model_path = "../../huggingface/llama-2-7b-chat-hf"
    elif chat_model == "qwen7b":
        model_path = "../../huggingface/Qwen-7B-Chat"
    elif chat_model == "qwen72b":
        model_path = "../../huggingface/Qwen-72B-Chat"
    else:
        raise NotImplementedError(f"chat_model {chat_model} not implemented")
    if "llama" in chat_model:
        tokenizer = LlamaTokenizer.from_pretrained(model_path)
        model = LlamaForCausalLM.from_pretrained(model_path)
    elif "qwen" in chat_model:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True)
    else:
        raise NotImplementedError(f"chat_model {chat_model} not implemented")
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()


    if "without_search" in prompt_method:
        if not os.path.exists(f"./user_intent_data/{dataset}/{chat_model}/{prompt_method}/"):
            os.makedirs(f"./user_intent_data/{dataset}/{chat_model}/{prompt_method}/", exist_ok=True)
        input_file = f"./user_intent_data/{dataset}/{dataset}-{split}.jsonl"
        output_file = f"./user_intent_data/{dataset}/{chat_model}/{prompt_method}/{chat_model}ppl-{dataset}-{split}.jsonl"
    else:
        if not os.path.exists(f"./user_intent_data/{dataset}/{chat_model}/{search_engine}/{prompt_method}/"):
            os.makedirs(f"./user_intent_data/{dataset}/{chat_model}/{search_engine}/{prompt_method}/", exist_ok=True)
        if "bing" in search_engine:
            input_file = f"./user_intent_data/{dataset}/{search_engine}/{prompt_method}/{search_engine}-{dataset}-{split}.jsonl"
            output_file = f"./user_intent_data/{dataset}/{chat_model}/{search_engine}/{prompt_method}/{chat_model}ppl-{dataset}-{split}.jsonl"
        else:
            input_file = f"./user_intent_data/{dataset}/{search_engine}/{prompt_method}/{rerank_model}-{search_engine}-{dataset}-{split}.jsonl"
            output_file = f"./user_intent_data/{dataset}/{chat_model}/{search_engine}/{prompt_method}/{rerank_model}-{chat_model}ppl-{dataset}-{split}.jsonl"

    with open(output_file, "w", encoding="utf-8") as f:
        pass
    with open(input_file, "r", encoding="utf-8") as f:
        data_lines = [json.loads(line) for line in f]

    if args.mini_dataset:
        data_lines = data_lines[:20]
    #  trim dataset that is too large
    data_lines = data_lines[:5000]
    print(f"dataset: {dataset}, split: {split}, prompt_method: {prompt_method}, chat_model: {chat_model}, search_engine: {search_engine}, rerank_model: {rerank_model}")
    if dataset in ["webglm-qa", "dolly", "eli5"]:
        llama_rag_prompt = llama_rag_prompt_fluency
        print(f"using long answer dataset {dataset}, change to fluency prompt for rag.")

    for idx, data_line in tqdm(enumerate(data_lines), total=len(data_lines),
                                   desc=f"dataset: {dataset}, split: {split}"):
        #  do search with vanilla query
        if prompt_method == "without_search":
            if "baichuan" in chat_model:
                model_input = "<C_Q> " + data_line["question"] + " <C_A>"
            elif "llama" in chat_model:
                model_input = "<s>[INST] " + data_line["question"] + " [/INST]"
            elif "qwen" in chat_model:
                model_input = data_line["question"]
            else:
                raise NotImplementedError(f"chat model {chat_model} not implemented")
        else:
            raise NotImplementedError(f"prompt method {prompt_method} not implemented")

        if idx == 0:
            print(f"model input: {model_input}")
            print(f"answer: {data_line['answer']}")

        #  perplexity inference
        ppl = cal_ppl(model_input, data_line["answer"])
        data_line["ppl"] = ppl
    with open(output_file, "a", encoding="utf-8") as f:
        for data_line in data_lines:
            f.write(json.dumps(data_line) + "\n")
