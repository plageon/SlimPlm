#!/bin/bash

# This script is used to run inference on a query rewrite model.
rewrite_model="v0114"
datasets=("asqa" "nq" "trivia-qa" "musique" "eli5")
#datasets=("asqa" )
splits=("test")
answer_model="qwen72b"

for dataset in "${datasets[@]}";
do
    for split in "${splits[@]}";
    do
        python SKR/icl.py \
            --dataset $dataset \
            --split $split \
            --inference_url "http://172.17.38.251:80" \
            --rewrite_model $rewrite_model \
            --answer_model $answer_model
    done
done

#        --provide_without_search_answer \