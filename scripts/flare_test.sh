#!/bin/bash
#export CUDA_VISIBLE_DEVICES=4,5,6,7
# This script is used to run inference on flare baseline
#datasets=("asqa" "nq" "trivia-qa" "musique" "eli5")
datasets=("asqa")
splits=("test")
chat_model="llama70b"

for dataset in "${datasets[@]}";
do
    for split in "${splits[@]}";
    do
        for round in {1..1};
        do
            python flare/flare_search.py \
            --mini_dataset \
                --dataset $dataset \
                --split $split \
                --chat_model $chat_model \
                --round $((round-1)) \
                --search_engine "kiltbm25"

#            python flare/flare_chat_inference.py \
#                --dataset $dataset \
#                --split $split \
#                --chat_model $chat_model \
#                --inference_url "http://llama-70b-chat.test.hongxin.bc-inner.com:32680/generate" \
#                --round $round
        done
    done
done