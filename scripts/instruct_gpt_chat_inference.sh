#!/bin/bash

# This script is used to run inference on a trained model.
#datasets=("asqa" "hotpot-qa" "nq" "trivia-qa" "2wiki" "musique" "eli5")
datasets=("hotpot-qa" "nq" "trivia-qa" "2wiki" "musique" "eli5")
splits=("test")
#prompt_methods=("without_search" "vanilla_search" "gpt4_rewrite_search" "v1225_rewrite_search" "rule_rewrite_search")
prompt_methods=("without_search" "gpt4_rewrite_search")
search_engine="kiltbm25"
chat_model="gpt35instruct"

for dataset in "${datasets[@]}";
do
    for split in "${splits[@]}";
    do
        for prompt_method in "${prompt_methods[@]}";
        do
            python user_intent/instruct_gpt_chat_inference.py \
                --dataset $dataset \
                --split $split \
                --chat_model $chat_model \
                --search_engine $search_engine \
                --prompt_method $prompt_method
        done
    done
done
