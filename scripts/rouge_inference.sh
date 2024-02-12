#!/bin/bash

# This script is used to run inference to calculate perplexity on a trained model.
#datasets=("eli5" "dolly")
datasets=("eli5")
splits=("trainfew")

#prompt_methods=("without_search" "vanilla_search" "gpt4_rewrite_search" "v1225_rewrite_search" "rule_rewrite_search")
prompt_method="without_search"
search_engine="kiltbm25"
chat_model="qwen72b"

for dataset in "${datasets[@]}";
do
    for split in "${splits[@]}";
    do
        python user_intent/rouge_inference.py \
            --dataset $dataset \
            --split $split \
            --chat_model $chat_model \
            --search_engine $search_engine \
            --prompt_method $prompt_method
    done
done
