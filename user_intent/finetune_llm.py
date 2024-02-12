import json
import logging
import os
import sys
from dataclasses import field, dataclass
from typing import List, Dict, Union, Any, Optional, Tuple

import torch
import transformers

from torch.nn import Module
from torch.utils.data import Dataset
from transformers import LlamaTokenizerFast, LlamaConfig, LlamaForCausalLM, HfArgumentParser, TrainingArguments, \
    PreTrainedTokenizerBase, GenerationConfig
from transformers.trainer_pt_utils import nested_detach
from transformers.utils import PaddingStrategy
import evaluate

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

logger = logging.getLogger(__name__)

mini_dataset_size = 32


dataset_tasks = {
    "asqa": "MutipleShort",
    "nq": "MutipleShort",
    "trivia-qa": "MutipleShort",
    "2wiki": "SingleShort",
    "musique": "SingleShort",
    "eli5": "Long",
}


@dataclass
class DataArguments:
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )

    dataset: str = field(
        default='', metadata={"help": "Dataset to use"}
    )
    task_name: str = field(
        default='', metadata={"help": "The name of the task to train on: "}
    )
    data_path: str = field(
        default='', metadata={"help": "The input data dir. Should contain the .jsonl files for the task."}
    )
    max_query_length: int = field(
        default=512, metadata={"help": "The maximum length of the query"}
    )
    max_target_length: int = field(
        default=512, metadata={"help": "The maximum length of the target"}
    )
    #  use small dataset for debugging
    mini_dataset: bool = field(
        default=False, metadata={"help": "Whether to use small dataset for debugging"}
    )
    predict_task: bool = field(
        default=False, metadata={"help": "Whether to predict task"}
    )
    predict_search_tags: bool = field(
        default=False, metadata={"help": "Whether to use search tags"}
    )
    provide_without_search_answer: bool = field(
        default=False, metadata={"help": "Whether to provide answers without search"}
    )
    predict_questions: bool = field(
        default=False, metadata={"help": "Whether to predict questions"}
    )
    predict_claims: bool = field(
        default=False, metadata={"help": "Whether to predict claims"}
    )


@dataclass
class ModelArgs:
    model_name_or_path: str = field(
        default='../../huggingface/llama-2-7b-chat-hf', metadata={"help": "The name or path of the model to use."}
    )
    num_virtual_tokens: int = field(
        default=15, metadata={"help": "The number of virtual tokens to use."}
    )
    prompt_init_text: str = field(
        default='', metadata={"help": "The initial text of the prompt."}
    )
    # resume_checkpoint_path: str = field(
    #     default='',
    #     metadata={"help": "Path to save the checkpoints"}
    # )


class QueryWriteTuningDataset(Dataset):
    def __init__(self, data_file, split, config, tokenizer=None):
        self.config = config
        self.data_file = data_file
        self.split = split
        self.tokenizer = tokenizer
        self.data = self._build_examples(data_file)
        logger.info(f"Loaded {len(self.data)} examples from {data_file}")

    def _build_examples(self, data_file):
        json_lines = [json.loads(line) for line in open(data_file, 'r', encoding='utf-8').readlines()]
        data_items = []
        key_without_search_answer = ""
        key_claims = ""
        if self.config.provide_without_search_answer:
            #  find key name ending with _without_search_answer
            key_without_search_answer = [key for key in json_lines[0].keys() if key.endswith("_without_search_answer")][
                0]
            rank = torch.distributed.get_rank()
            if rank == 0:
                print(f"key_without_search_answer: {key_without_search_answer}")
        if self.config.predict_claims and "query-rewrite" in self.config.task_name:
            key_claims = [key for key in json_lines[0].keys() if key.endswith("_claims")][0]
            rank = torch.distributed.get_rank()
            if rank == 0:
                print(f"key_claim: {key_claims}")
        for idx, json_line in enumerate(json_lines):
            # if "query-rewrite" in self.config.task_name:
            if self.config.provide_without_search_answer:
                query = (f"<s>[INST] <<SYS>>\nYou are a helpful assistant. Your task is to parse user input into"
                         f" structured formats according to the coarse answer. Current datatime is 2023-12-20 9:47:28"
                         f" <</SYS>>\n Course answer: (({json_line[key_without_search_answer]}))\nQuestion: "
                         f"(({json_line['question']})) [/INST]")
            else:
                query = (f"<s>[INST] <<SYS>>\nYou are a helpful assistant. Your task is to parse user input into"
                         f" structured formats. Current datatime is 2023-12-20 9:47:28"
                         f" <</SYS>>\n{json_line['question']} [/INST]")
            # elif "search-tag" in self.config.task_name:
            #     if self.config.provide_without_search_answer:
            #         query = (f"<s>[INST] <<SYS>>\nYou are a helpful assistant. Your task is to judge whether the model"
            #                  f" has known the information about the question according to the coarse answer."
            #                  f" <</SYS>>\n Course answer: (({json_line[key_without_search_answer]}))\nQuestion: "
            #                  f"(({json_line['question']})) [/INST]")
            #     else:
            #         query = (f"<s>[INST] <<SYS>>\nYou are a helpful assistant. Your task is to parse user input into"
            #                  f" structured formats. Current datatime is 2023-12-20 9:47:28"
            #                  f" <</SYS>>\n{json_line['question']} [/INST]")
            # else:
            #     raise ValueError(f"task_name {self.config.task_name} is not supported")
            query_inputs = self.tokenizer(query, truncation=True, padding=True, max_length=self.config.max_query_length,
                                          add_special_tokens=False)
            task_str = ""
            timeliness_str = ""
            if self.config.predict_task and "query-rewrite" in self.config.task_name:
                #  query-rewrite task
                if "query-rewrite" in self.config.task_name:
                    task_str = f"<Task({json_line['task']})> "
                    timeliness_str = "<Timeliness(True)> " if json_line["timeliness"] else "<Timeliness(False)> "
                #  search-tag task
                # elif "search-tag" in self.config.task_name:
                #     task_str = f"<Task({dataset_tasks[json_line['dataset']]})> <Dataset({json_line['dataset']})>"
                #     timeliness_str = ""
                else:
                    raise ValueError(f"task_name {self.config.task_name} is not supported")
            search_str = ""
            if self.config.predict_search_tags and "search-tag" in self.config.task_name:
                if "known" in json_line:
                    search_str += "<Known(True)> " if json_line["known"] else "<Known(False)> "
                if "search" in json_line:
                    search_str += "<Search(True)> " if json_line["search"] else "<Search(False)> "
            if self.config.predict_questions and "query-rewrite" in self.config.task_name:
                question_str = "<Questions> "
                for question in json_line["questions"]:
                    question_str += f"<Question({question['question']})> "
                    question_str += f"<NeedSearch({question['needSearch']})> "
                    question_str += f"<Query({question['searchWord']})> " if "searchWord" in question else "<Query()> "

                question_str += "</Questions> "
            else:
                question_str = ""
            if self.config.predict_claims and "query-rewrite" in self.config.task_name:

                claim_str = "<Claims> "
                for claim in json_line[key_claims]:
                    claim_str += f"<Claim({claim['claim']})> "
                    claim_str += f"<NeedSearch({claim['needSearch']})> "
                    claim_str += f"<Query({claim['query']})> "
                claim_str += "</Claims> "
            else:
                claim_str = ""

            #  if process rank is 0, print the first example

            #  use label feature to determine whether to do retrieval or not
            target_text = task_str + timeliness_str + search_str + question_str + claim_str + "</s>"
            try:
                rank = torch.distributed.get_rank()
                if rank == 0:
                    if idx == 0:
                        print(f"query: {query}")
                        print(f"target_text: {target_text}")
            except Exception as e:
                print(f"output data sample failed: {e}")

            target_inputs = self.tokenizer(
                target_text, truncation=True, padding=True, max_length=self.config.max_target_length,
                add_special_tokens=False)
            # concat query_inputs["input_ids"] and target_inputs["input_ids"]
            query_input_ids = query_inputs["input_ids"] + target_inputs["input_ids"]

            # add -100 equal to the length of query to target_inputs["input_ids"], and pad_token_id to the end
            target_input_ids = len(query_inputs["input_ids"]) * [-100] \
                               + target_inputs["input_ids"]

            #  label contains the retrieval label and the generation label
            data_item = {
                "input_ids": query_input_ids,
                "attention_mask": [1] * len(query_input_ids),
                "labels": target_input_ids,
                # "binary_labels": [target_text2id[target_text]],
            }
            data_items.append(data_item)
        return data_items

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


@dataclass
class DataCollatorWithPadding:
    """
    Data collator that will dynamically pad the inputs received.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            - `True` or `'longest'` (default): Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'`: No padding (i.e., can output a batch with sequences of different lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # concat features together
        features = {k: [feature[k] for feature in features] for k in features[0]}
        batch_size = len(features["input_ids"])
        max_length = max(len(feature) for feature in features["input_ids"])
        for i in range(batch_size):
            required_input = features["input_ids"][i]
            difference = max_length - len(required_input)
            #  padding side is left
            features["input_ids"][i] = [self.tokenizer.bos_token_id] * difference + required_input
            features["attention_mask"][i] = [0] * difference + features["attention_mask"][i]
            features["labels"][i] = [-100] * difference + features["labels"][i]
            # features["binary_labels"][i] = [0] * difference + features["binary_labels"][i]

        batch = {
            "input_ids": torch.tensor(features["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(features["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(features["labels"], dtype=torch.long),
            # "binary_labels": torch.tensor(features["binary_labels"], dtype=torch.long),
        }
        return batch


class QueryWriteTrainer(transformers.Trainer):
    def __init__(self, max_generation_length, **kwargs):
        super().__init__(**kwargs)
        self.max_generation_length = max_generation_length
        generation_configs = GenerationConfig.from_model_config(self.model.config)

        self.model.config.update({
            "max_new_tokens": max_generation_length,
            "num_beams ": 1,
            "do_sample": False,
        })
        if self.is_world_process_zero():
            print(f"generation_configs: {generation_configs}")
        self.generate_config = generation_configs

    def _prediction_step(
            self,
            model: Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
            loss = loss.mean().detach()

            if isinstance(outputs, dict):
                logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"])
            else:
                logits = outputs[1:]

            #  collate inputs for generation
            input_ids = []
            attention_mask = []
            max_input_length = 0
            batch_size = inputs["input_ids"].shape[0]
            for idx in range(batch_size):
                #  remove labels from inputs, length is self.max_generation_length
                label_length = torch.sum(inputs["labels"][idx] != -100)
                max_input_length = max(max_input_length, label_length)
                input_ids.append(inputs["input_ids"][idx][:label_length])
            for idx in range(batch_size):
                input_ids[idx] = torch.cat((torch.ones(max_input_length - len(input_ids[idx]), dtype=torch.long)
                                            .to(torch.cuda.current_device()) * self.tokenizer.bos_token_id,
                                            input_ids[idx]), dim=0)
                attention_mask.append(torch.tensor([0] * (max_input_length - len(input_ids[idx]))
                                                   + [1] * len(input_ids[idx]), dtype=torch.long).to(
                    torch.cuda.current_device()))
            input_ids = torch.stack(input_ids, dim=0)
            attention_mask = torch.stack(attention_mask, dim=0)
            input_length = input_ids.shape[1]

            #  if models type is dataparallel, then model.module.generate
            if isinstance(model, torch.nn.DataParallel):
                model = model.module

            #  use greedy decoding
            generate_output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_scores=True,
                return_dict_in_generate=True,
            )

        logits = nested_detach(logits)
        logits = logits[0]
        gen_logits = generate_output.sequences[:, input_length:]
        # if self.is_world_process_zero():
        #     print(f"gen_logits_size: {gen_logits.size()}")

        labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
        if len(labels) == 1:
            labels = labels[0]

        return (loss, gen_logits, labels)

    def _predict(
            self, test_dataset: Dataset, ignore_keys: Optional[List[str]] = None, metric_key_prefix: str = "test"
    ) -> Optional[List]:
        pred_output = super().predict(test_dataset, ignore_keys, metric_key_prefix)
        #  use numpy argmax to get the predicted labels
        pred_logits = pred_output.predictions
        #  remove overflows from the generation results
        #  replace -100s with self.tokenizer.pad_token_id
        pred_logits[pred_logits == -100] = self.tokenizer.pad_token_id
        output_strs = [self.tokenizer.decode(output, skip_special_tokens=True) for output in pred_logits]

        #  remove -100s at the beginning of the labels, labels are padded with -100s, labels are numpy arrays
        return output_strs


def get_dataset(data_args, tokenizer):
    assert data_args.dataset in data_args.data_path, \
        f"dataset {data_args.dataset} is not in the data_path {data_args.data_path}"
    data = {}
    file_list = os.listdir(data_args.data_path)
    for split in ['train', 'val', 'test']:
        #  test_gpt4_dolly.jsonl'
        file_name = [file for file in file_list if split in file][0]
        gpt4_result_file = os.path.join(data_args.data_path, file_name)
        data[split] = QueryWriteTuningDataset(
            data_file=gpt4_result_file,
            split=split,
            config=data_args,
            tokenizer=tokenizer,
        )
    if data_args.mini_dataset:
        return data['train'][:mini_dataset_size], data['val'][:mini_dataset_size], data['test'][:mini_dataset_size]
    else:
        return data['train'], data['val'], data['test']


def query_rewrite_finetuning():
    parser = HfArgumentParser((DataArguments, ModelArgs, TrainingArguments))
    data_args, model_args, training_args = parser.parse_args_into_dataclasses()
    rank = torch.distributed.get_rank()
    if rank == 0:
        transformers.utils.logging.set_verbosity_debug()
        transformers.generation.configuration_utils.logger.setLevel(logging.WARNING)
    else:
        transformers.utils.logging.set_verbosity_error()
        transformers.generation.configuration_utils.logger.setLevel(logging.ERROR)
    n_gpu = training_args.n_gpu

    model_name_or_path = model_args.model_name_or_path
    tokenizer = LlamaTokenizerFast.from_pretrained(model_name_or_path)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.bos_token_id

    train_dataset, val_dataset, test_dataset = get_dataset(data_args, tokenizer)

    rouge = evaluate.load("./evaluate_utils/rouge/")

    #  define the compute metrics function
    def compute_metrics(pred):
        #  decode the generation results
        pred_logits = pred.predictions
        labels = pred.label_ids
        #  remove overflows from the generation results
        pred_logits[pred_logits == -100] = tokenizer.pad_token_id
        output_strs = [tokenizer.decode(output, skip_special_tokens=True) for output in pred_logits]

        #  remove -100s at the beginning of the labels, labels are padded with -100s, labels are numpy arrays
        labels = [label[label != -100] for label in labels]
        label_strs = [tokenizer.decode(label, skip_special_tokens=True) for label in labels]
        #  check task correctness
        #  extract <Task(xxx)> from label_strs and output_strs
        label_task = [label_str.split("<Task(")[1].split(")>")[0] for label_str in label_strs]
        #  extract <Task(xxx)> may not present in output_strs
        output_task = []
        for output_str in output_strs:
            try:
                output_task.append(output_str.split("<Task(")[1].split(")>")[0])
            except:
                output_task.append("")
        #  check whether the task is correct
        task_exact_match = [label_task[idx].lower() == output_task[idx].lower() for idx in range(len(label_task))]
        task_exact_match = sum(task_exact_match) / len(task_exact_match)

        #  check timeliness correctness
        #  extract <Timeliness(True)> or <Timeliness(False)> from label_strs and output_strs
        label_timeliness = [label_str.split("<Timeliness(")[1].split(")>")[0] for label_str in label_strs]
        #  extract <Timeliness(xxx)> may not present in output_strs
        output_timeliness = []
        for output_str in output_strs:
            try:
                output_timeliness.append(output_str.split("<Timeliness(")[1].split(")>")[0])
            except:
                output_timeliness.append("")
        #  check whether the timeliness is correct
        timeliness_exact_match = [label_timeliness[idx].lower() == output_timeliness[idx].lower() for idx in
                                  range(len(label_timeliness))]
        timeliness_exact_match = sum(timeliness_exact_match) / len(timeliness_exact_match)

        #  check question rouge score
        #  extract <Questions> xxx </Questions> from label_strs and output_strs
        label_questions = [label_str.split("<Questions>")[1].split("</Questions>")[0] for label_str in label_strs]
        #  extract <Questions> xxx </Questions> may not present in output_strs
        output_questions = []
        for output_str in output_strs:
            try:
                output_questions.append(output_str.split("<Questions>")[1].split("</Questions>")[0])
            except IndexError:
                output_questions.append("")
        #  calculate the rouge score
        rouge_results = rouge.compute(predictions=output_questions, references=label_questions)

        metrics = {
            "task_exact_match": task_exact_match,
            "timeliness_exact_match": timeliness_exact_match,
            "rouge1": rouge_results["rouge1"],
            "rouge2": rouge_results["rouge2"],
            "rougeL": rouge_results["rougeL"],
            "rougeLsum": rouge_results["rougeLsum"],
        }
        return metrics

    if not training_args.resume_from_checkpoint:
        model = LlamaForCausalLM.from_pretrained(model_name_or_path)
    else:
        model = LlamaForCausalLM.from_pretrained(training_args.resume_from_checkpoint)

    if model.config.pad_token_id is None:
        model.config.pad_token_id = model.config.bos_token_id

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = QueryWriteTrainer(
        max_generation_length=data_args.max_target_length,
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        # compute_metrics=compute_metrics,
        data_collator=data_collator,
    )

    if training_args.do_train:
        #  train the model
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        # only the main process save the model
        if trainer.is_world_process_zero():
            trainer.save_model()

    # evaluate the model
    predictions_dir = f'predictions/{data_args.task_name}/'
    if not os.path.exists(os.path.dirname(predictions_dir)):
        os.makedirs(os.path.dirname(predictions_dir), exist_ok=True)

    test_datasets = {'val': val_dataset, 'test': test_dataset}
    metrics = {}
    for split in ['val', 'test']:
        test_metrics = trainer.evaluate(eval_dataset=test_datasets[split])
        # predicted_str = trainer.predict(test_dataset=test_datasets[split])
        # # merge the predicted labels and datalines
        # data_file = os.path.join(data_args.data_path, f"gpt4-{data_args.dataset}-{split}.jsonl")
        #
        # json_lines = [json.loads(line) for line in open(data_file, 'r', encoding='utf-8').readlines()]
        #
        # if data_args.mini_dataset:
        #     json_lines = json_lines[:mini_dataset_size]
        # with open(predictions_file, 'w', encoding='utf-8') as f:
        #     for idx, data_line in enumerate(json_lines):
        #         data_line["preds"]= predicted_str[idx]
        #         for metric in ["task_exact_match", "timeliness_exact_match", "rouge1", "rouge2", "rougeL"]:
        #             #  eval_ is the prefix of the metrics
        #             data_line[metric] = test_metrics['eval_' + metric]
        #         f.write(json.dumps(data_line, ensure_ascii=False) + '\n')

        # save the test result
        metrics[split] = test_metrics

    if trainer.is_world_process_zero():
        for split in ['val', 'test']:
            metrics_file = f'{predictions_dir}/{split}_split_metrics.json'
            predictions_file = f'{predictions_dir}/{split}_split_predictions.jsonl'
            with open(metrics_file, 'w', encoding='utf-8') as f:
                json.dump(metrics[split], f, indent=4, ensure_ascii=False)
            # display the test result

            print(f"-------------------{split}-------------------")
            for key in metrics[split]:
                print(f"{split} {key}", metrics[split][key])


if __name__ == "__main__":
    query_rewrite_finetuning()
