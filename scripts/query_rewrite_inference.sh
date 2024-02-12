#!/bin/bash
#export CUDA_VISIBLE_DEVICES=4
# This script is used to run inference on a query rewrite model.
rewrite_model="v0104"
datasets=("asqa" "nq" "trivia-qa" "musique" "eli5")
#datasets=("eli5")
splits=("test")
answer_model="phi2"

for dataset in "${datasets[@]}";
do
    for split in "${splits[@]}";
    do
        python user_intent/query_rewrite_inference.py \
        --local_inference \
            --provide_without_search_answer \
            --dataset $dataset \
            --split $split \
            --address "172.17.99.53:80" \
            --language "en" \
            --rewrite_model $rewrite_model \
            --answer_model $answer_model
    done
done

#        --provide_without_search_answer \
#        --local_inference \