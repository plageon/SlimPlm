#!/bin/bash

# This script is used to run inference on a trained model.
#datasets=("asqa" "hotpot-qa" "nq" "trivia-qa")
datasets=("2wiki" "musique" "eli5")
splits=("trainfew")
#prompt_methods=("without_search" "vanilla_search" "gpt4_rewrite_search" "v1221_rewrite_search" "rule_rewrite_search")
prompt_methods=("without_search")
search_engine="kiltbm25"
chat_model="llama13b"
sep_model="gpt4"

for dataset in "${datasets[@]}";
do
    for split in "${splits[@]}";
    do
        for prompt_method in "${prompt_methods[@]}";
        do
            python user_intent/seperate_claims.py \
                --dataset $dataset \
                --split $split \
                --chat_model $chat_model \
                --sep_model $sep_model \
                --search_engine $search_engine \
                --prompt_method $prompt_method
            wc "user_intent_data/${dataset}/${chat_model}/${prompt_method}/unparsed-${sep_model}sep-${chat_model}-${dataset}-${split}.jsonl"
        done
    done
done


