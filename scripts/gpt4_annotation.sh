#!/bin/bash

# This script is used to use gpt4 api to annotate the data
#datasets=("asqa" "eli5" "hotpot-qa" "nq")
datasets=("2wiki" "musique")
splits=("trainfew")

for dataset in "${datasets[@]}";
do
    for split in "${splits[@]}";
    do
        python user_intent/gpt4_annotation.py \
            --dataset $dataset \
            --split $split
    done
done