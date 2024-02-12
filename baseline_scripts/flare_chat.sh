#!/bin/bash
#export CUDA_VISIBLE_DEVICES=4,5,6,7
# This script is used to run inference on flare baseline
datasets=("asqa" "nq" "trivia-qa" "musique" "eli5")
#datasets=("trivia-qa" "musique" "eli5")
splits=("test")
chat_model="qwen72b"
round=3

for dataset in "${datasets[@]}";
do
    for split in "${splits[@]}";
    do
        python flare/flare_chat_inference.py \
            --dataset $dataset \
            --split $split \
            --chat_model $chat_model \
            --inference_url "http://llama-70b-chat.test.hongxin.bc-inner.com:32680/generate" \
            --round $round
    done
done