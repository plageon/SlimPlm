#!/bin/bash

# This script is used to run inference to calculate perplexity on a trained model.
datasets=("asqa" "hotpot-qa" "nq" "trivia-qa" "2wiki" "musique" "qampari")
#datasets=("asqa")
splits=("test" "trainfew")
#splits=("test")
answer_model="llama7b"
judge_model="v0116"

for dataset in "${datasets[@]}";
do
    for split in "${splits[@]}";
    do
        python user_intent/necessity_retrieval_inference.py \
            --provide_without_search_answer \
            --dataset $dataset \
            --split $split \
            --answer_model $answer_model \
            --judge_model $judge_model
    done
done
