#!/bin/bash
#export CUDA_VISIBLE_DEVICES=4,5,6,7
# This script is used to run inference on flare baseline
#datasets=("asqa" "nq" "trivia-qa" "musique" "eli5")
datasets=("eli5")
splits=("test")
chat_model="llama70b"
round=2

for dataset in "${datasets[@]}";
do
    for split in "${splits[@]}";
    do
        python itergen/itergen_chat.py \
            --dataset $dataset \
            --split $split \
            --chat_model $chat_model \
            --inference_url "http://172.17.127.31:80" \
            --round $round
    done
done