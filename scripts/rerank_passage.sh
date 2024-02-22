#!/bin/bash
#export CUDA_VISIBLE_DEVICES=0
# This script is used to apply search pipeline
#datasets=("asqa" "nq" "trivia-qa" "musique" "eli5")
datasets=("eli5")
splits=("test")
#search_methods=("v0119_rewrite_search" "v0118llama7b_rewrite_search" "v0118llama7bv0104_rewrite_search" )
#search_methods=("v0118qwen7bv0104_rewrite_search" "v0118tinyllamav0104_rewrite_search" "v0118phi2v0104_rewrite_search")
search_methods=( "v0118llama7bv0104_rewrite_search")
search_engine="kiltbm25"

for dataset in "${datasets[@]}";
do
    for split in "${splits[@]}";
    do
        for search_method in "${search_methods[@]}";
        do
            python search_utils/rerank_passage.py \
                --dataset $dataset \
                --split $split \
                --rerank_model "e5base" \
                --search_method $search_method \
                --search_engine $search_engine
        done
    done
done
