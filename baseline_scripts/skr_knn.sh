#!/bin/bash

# This script is used to run inference on a query rewrite model.
judge_model="v01130"
#datasets=("asqa" "hotpot-qa" "nq" "trivia-qa" "2wiki" "musique" "qampari" "dolly" "eli5")
datasets=("asqa" "nq" "trivia-qa" "musique" "eli5")
splits=("test")
answer_model="llama70b"

for dataset in "${datasets[@]}";
do
    for split in "${splits[@]}";
    do
        python SKR/skr_knn.py \
            --dataset $dataset \
            --split $split \
            --judge_model $judge_model \
            --answer_model $answer_model
    done
done