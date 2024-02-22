#!/bin/bash

# This script is used to apply search pipeline
datasets=("asqa" "nq" "trivia-qa" "musique" "eli5")
#datasets=("eli5")
splits=("test")
#search_methods=("vanilla_search" "gpt4_rewrite_search" "v1221_rewrite_search" "rule_rewrite_search" "gpt4plusgpt4sep_rewrite_search")
search_methods=("v0118llama7b_rewrite_search" "v0118llama7bv0104_rewrite_search" )
#search_methods=("v0118qwen7bv0104_rewrite_search" "v0118tinyllamav0104_rewrite_search" "v0118phi2v0104_rewrite_search")
#search_methods=("v0118llama7bv0104_rewrite_search")
search_engine="kiltbm25"
address="172.17.69.235:5050"

for dataset in "${datasets[@]}";
do
    for split in "${splits[@]}";
    do
        for search_method in "${search_methods[@]}";
        do
            if [ "$search_engine" == "bing" ]; then
                python search_utils/search_pipeline_apply.py \
                    --dataset $dataset \
                    --split $split \
                    --search_method $search_method
            elif [ "$search_engine" == "kiltbm25" ]; then
                 python search_utils/kiltbm25.py \
                      --address $address \
                      --dataset $dataset \
                      --split $split \
                      --search_method $search_method \
                      --search_engine $search_engine
            fi
        done
    done
done
