#!/bin/bash

# This script is used to extract web contents
datasets=("asqa")
splits=("val")

for dataset in "${datasets[@]}";
do
    for split in "${splits[@]}";
    do
        python search_utils/extract_web_contents.py \
            --dataset $dataset \
            --split $split \
            --search_method "gpt4_rewrite_search"
    done
done
