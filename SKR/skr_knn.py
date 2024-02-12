import argparse
import json
import torch
import transformers
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
import faiss
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

model_path = "../../huggingface/unsup-simcse-bert-base-uncased"


class SentRetriever(object):
    def __init__(self, input_path, output_path):
        self.input_sents_file = input_path
        self.output_sents_file = output_path
        self.batch_size = 64

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path).to(self.device)
        self.model.eval()

        self.query_model = AutoModel.from_pretrained(model_path).to(
            self.device)
        self.query_tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.query_model.eval()

        if os.path.isfile(self.output_sents_file):
            self.sent_features, self.sent_set, self.status_set, self.passage_set, self.cot_set = self.load_sent_features(
                self.output_sents_file)
            print(f"{len(self.sent_set)} sents loaded from {self.output_sents_file}")
        else:
            self.sent_set, self.status_set, self.passage_set, self.cot_set = self.load_sents()
            self.sent_features = self.build_sent_features()
            self.save_sent_features(self.sent_features, self.sent_set, self.status_set, self.passage_set, self.cot_set,
                                    self.output_sents_file)

    def load_sents(self):
        i_set = []
        passage_set = []
        status_set = []
        cot_set = []
        test_file = open(self.input_sents_file, 'r', encoding='utf-8')
        test_data = [json.loads(line) for line in test_file]
        for element in test_data:
            # if 'Question' in element.keys():
            #     if element['status'] == 'same':
            #         continue
            #     i_set.append(element['Question'])
            #     status_set.append(element['status'])
            #     passage_set.append([])
            #     cot_set.append([])
            i_set.append(element['question'])
            status_set.append(element['known'])
            passage_set.append([])
            cot_set.append([])

        print(f"Loading {len(i_set)} sents in total.")
        return i_set, status_set, passage_set, cot_set

    def build_sent_features(self):
        print(f"Build features for {len(self.sent_set)} sents...")
        batch_size, counter = self.batch_size, 0
        batch_text = []
        all_i_features = []
        for i_n in tqdm(self.sent_set):
            counter += 1
            batch_text.append(i_n)
            if counter % batch_size == 0 or counter >= len(self.sent_set):
                with torch.no_grad():
                    i_input = self.tokenizer(batch_text, return_tensors="pt", padding=True, truncation=True,
                                             max_length=128).to(self.device)
                    i_feature = self.model(**i_input, output_hidden_states=True, return_dict=True).pooler_output
                i_feature /= i_feature.norm(dim=-1, keepdim=True)
                all_i_features.append(i_feature.squeeze().to('cpu'))
                batch_text = []
        returned_text_features = torch.cat(all_i_features)
        return returned_text_features

    def save_sent_features(self, sent_feats, sent_names, sent_status, sent_passages, sent_cots, path_to_save):
        assert len(sent_feats) == len(sent_names)
        print(f"Save {len(sent_names)} sent features at {path_to_save}...")
        torch.save({'sent_feats': sent_feats, 'sent_names': sent_names, 'sent_status': sent_status,
                    'sent_passages': sent_passages, 'sent_cots': sent_cots}, path_to_save)
        print(f"Done.")

    def load_sent_features(self, path_to_save):
        print(f"Load sent features from {path_to_save}...")
        checkpoint = torch.load(path_to_save)
        return checkpoint['sent_feats'], checkpoint['sent_names'], checkpoint['sent_status'], checkpoint[
            'sent_passages'], checkpoint['sent_cots']

    def get_text_features(self, text):
        self.query_model.eval()
        with torch.no_grad():
            i_input = self.query_tokenizer(text, return_tensors="pt").to(self.device)
            text_features = self.query_model(**i_input, output_hidden_states=True, return_dict=True).pooler_output
        text_features /= text_features.norm(dim=-1, keepdim=True)

        return text_features

    def setup_faiss(self):
        self.index = faiss.IndexFlatIP(768)  # BERT-base dimension
        self.index.add(self.sent_features.numpy())  #

    def faiss_retrieve(self, text, topk=5):
        text_f = self.get_text_features(text)
        D, I = self.index.search(text_f.cpu().numpy(), topk)
        return D, I


def run_gpt3(test_path, output_path, topk):
    test_file = open(test_path, 'r')
    test_data = [json.loads(line) for line in test_file]
    for idx, test_case in tqdm(enumerate(test_data), total=len(test_data), desc=f"Running {dataset} {split}"):
        # print(idx, len(test_data))
        question = test_case['question'].strip()
        test_case[f'{judge_model}_rewrite'] = {}
        train_skr_prop = [47, 62, 740]
        ir_not = ir = 0
        D, I = sentRetriever.faiss_retrieve(question, topk)
        for idx_ in I[0]:
            #  known is False, which means the question is unknown, and ir is better
            if sentRetriever.status_set[idx_] == False:
                ir += 1
            #  known is True, which means the question is known, and ir is worse
            elif sentRetriever.status_set[idx_] == True:
                ir_not += 1
            else:
                pass
        if train_skr_prop[0] < train_skr_prop[1]:
            if ir_not > ir and (ir_not - ir) >= int(
                    (train_skr_prop[1] - train_skr_prop[0]) * topk / sum(train_skr_prop[:3])):
                test_case[f'{judge_model}_rewrite']["known"] = True
            else:
                test_case[f'{judge_model}_rewrite']["known"] = False
        else:
            if ir > ir_not and (ir - ir_not) >= int(
                    (train_skr_prop[0] - train_skr_prop[1]) * topk / sum(train_skr_prop[:3])):
                test_case[f'{judge_model}_rewrite']["known"] = False
            else:
                test_case[f'{judge_model}_rewrite']["known"] = True
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        for test_case in test_data:
            f.write(json.dumps(test_case) + '\n')


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--dataset", type=str, default="webglm-qa")
    arg_parser.add_argument("--split", type=str, default="test")
    arg_parser.add_argument("--mini_dataset", action="store_true", help="whether to use mini dataset")
    arg_parser.add_argument("--answer_model", type=str, default="llama13b")
    arg_parser.add_argument("--judge_model", type=str, default="v0113")
    args = arg_parser.parse_args()
    dataset = args.dataset
    split = args.split
    answer_model = args.answer_model
    judge_model = args.judge_model

    #  find data name ends with train.jsonl
    data = {}
    index_data_dir = f"./user_intent_data/mixed/{judge_model}"
    file_list = os.listdir(index_data_dir)
    # print(file_list)
    for split in ['train', 'val', 'test']:
        #  test_gpt4_dolly.jsonl'
        file_name = [file for file in file_list if split in file][0]
        gpt4_result_file = os.path.join(index_data_dir, file_name)
        data[split] = gpt4_result_file
    index_input_path = data['train']
    index_output_path = index_input_path.replace('.jsonl', '-index.pt')

    sentRetriever = SentRetriever(index_input_path, index_output_path)
    sentRetriever.setup_faiss()

    test_path = f"./user_intent_data/{dataset}/{answer_model}/without_search/{answer_model}-{dataset}-{split}.jsonl"
    output_path = f"./user_intent_data/{dataset}/rewrite/{judge_model}/{judge_model}-{dataset}-{split}.jsonl"
    top_k = 8
    run_gpt3(test_path, output_path, top_k)
