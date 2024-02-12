#!/bin/bash
#export CUDA_VISIBLE_DEVICES=4
# This script is used to run inference on a query rewrite model.
judge_model="v0104"
#datasets=("asqa" "nq" "trivia-qa" "musique" "eli5")
datasets=("eli5")
splits=("test")
search_method="v0118qwen7b_rewrite_search"

for dataset in "${datasets[@]}";
do
    for split in "${splits[@]}";
    do
        python user_intent/claim_query_filter.py \
            --local_inference \
            --dataset $dataset \
            --split $split \
            --inference_url "172.17.78.255:80" \
            --judge_model $judge_model \
            --search_method $search_method
    done
done

#            --local_inference \
