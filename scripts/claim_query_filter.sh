#!/bin/bash
#export CUDA_VISIBLE_DEVICES=4
# This script is used to run inference on a query rewrite model.
judge_model="v0104"
datasets=("asqa" "nq" "trivia-qa" "musique" "eli5")
#datasets=("trivia-qa")
splits=("test")
search_method="v0118llama7b_rewrite_search"

for dataset in "${datasets[@]}";
do
    for split in "${splits[@]}";
    do
        python user_intent/claim_query_filter.py \
            --dataset $dataset \
            --split $split \
            --inference_url "172.17.100.57:80" \
            --judge_model $judge_model \
            --search_method $search_method
    done
done

#            --local_inference \
