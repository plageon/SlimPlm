#!/bin/bash
#export CUDA_VISIBLE_DEVICES=4
# This script is used to run inference to calculate perplexity on a trained model.
#datasets=("asqa" "hotpot-qa" "nq" "trivia-qa" "2wiki" "musique" "qampari")
datasets=("2wiki" "musique")
splits=("trainmore")

#prompt_methods=("without_search" "vanilla_search" "gpt4_rewrite_search" "v1225_rewrite_search" "rule_rewrite_search")
prompt_method="without_search"
search_engine="kiltbm25"
chat_model="llama7b"

for dataset in "${datasets[@]}";
do
    for split in "${splits[@]}";
    do
        python user_intent/perplexity_inference.py \
            --dataset $dataset \
            --split $split \
            --chat_model $chat_model \
            --search_engine $search_engine \
            --prompt_method $prompt_method
    done
done
