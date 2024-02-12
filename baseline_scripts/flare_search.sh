#!/bin/bash
#export CUDA_VISIBLE_DEVICES=4,5,6,7
# This script is used to run inference on flare baseline
datasets=("asqa" "nq" "trivia-qa" "musique" "eli5")
#datasets=("nq" "musique")
splits=("test")
chat_model="qwen72b"
round=2

for dataset in "${datasets[@]}";
do
    for split in "${splits[@]}";
    do
        python flare/flare_search.py \
            --dataset $dataset \
            --split $split \
            --chat_model $chat_model \
            --round $round \
            --search_engine "kiltbm25"
    done
done