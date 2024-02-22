#!/bin/bash
#export CUDA_VISIBLE_DEVICES=4
# This script is used to run inference on a query rewrite model.
rewrite_model="v0104"
datasets=("asqa" "nq" "trivia-qa" "musique" "eli5")
#datasets=("nq" "trivia-qa")
splits=("test")
answer_model="llama7b"

for dataset in "${datasets[@]}";
do
    for split in "${splits[@]}";
    do
        python user_intent/query_rewrite_inference.py \
            --provide_without_search_answer \
            --dataset $dataset \
            --split $split \
            --address "172.17.54.3:80" \
            --language "en" \
            --rewrite_model $rewrite_model \
            --answer_model $answer_model
    done
done

#        --provide_without_search_answer \
#        --local_inference \