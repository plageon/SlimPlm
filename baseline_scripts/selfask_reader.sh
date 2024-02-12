#!/bin/bash
#export CUDA_VISIBLE_DEVICES=4,5,6,7
# This script is used to run inference on flare baseline
#datasets=("asqa" "nq" "trivia-qa" "musique" "eli5")
datasets=("musique")
splits=("test")
chat_model="qwen72b"
round=1

for dataset in "${datasets[@]}";
do
    for split in "${splits[@]}";
    do
        python selfask/selfask_reader.py \
            --dataset $dataset \
            --split $split \
            --chat_model $chat_model \
            --round $round \
            --inference_url "http://172.17.82.101:80"
    done
done