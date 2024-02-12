#!/bin/bash
#export CUDA_VISIBLE_DEVICES=4,5,6,7
# This script is used to run inference on flare baseline
#datasets=("asqa" "nq" "trivia-qa" "musique" "eli5")
datasets=("eli5")
splits=("test")
chat_model="llama70b"
round=1

for dataset in "${datasets[@]}";
do
    for split in "${splits[@]}";
    do
        python itergen/itergen_search.py \
            --dataset $dataset \
            --split $split \
            --chat_model $chat_model \
            --round $round \
            --search_api "http://172.17.31.12:5050/search/bm25_e5_rerank/" \
            --address "172.17.105.34" \
            --search_engine "kiltbm25"
    done
done