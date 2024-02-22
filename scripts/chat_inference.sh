#!/bin/bash
#export CUDA_VISIBLE_DEVICES=4,5,6,7
# This script is used to run inference on a trained model.
datasets=("asqa" "nq" "trivia-qa" "musique" "eli5")
#datasets=("musique" "eli5")
splits=("test")
#prompt_methods=("without_search" "vanilla_search" "gpt4_rewrite_search" "v1225_rewrite_search" "rule_rewrite_search")
#prompt_methods=("vanilla_search" "gpt4_rewrite_search" "v0119_rewrite_search" "v0118llama7b_rewrite_search" "v0118llama7bv0104_rewrite_search" "v0118qwen7bv0104_rewrite_search" "v0118tinyllamav0104_rewrite_search" "v0118phi2v0104_rewrite_search")
prompt_methods=("v0118llama7b_rewrite_search" "v0118llama7bv0104_rewrite_search")
search_engine="kiltbm25"
chat_model="qwen72b"

subString="qwen"
if [ $chat_model = "llama7b" ]; then
    inference_url="http://172.17.74.234:80"
elif [ $chat_model = "llama13b" ]; then
    inference_url="http://172.17.85.169:80"
elif [ $chat_model = "llama70b" ]; then
#    inference_url="http://llama-70b-chat.test.hongxin.bc-inner.com:32680/generate"
    inference_url="http://172.17.100.220:80"
elif [ $chat_model = "tinyllama" ]; then
    inference_url="http://172.17.48.203:80"
elif [ $chat_model = "phi2" ]; then
    inference_url="local"
elif [ $chat_model = "baichuan7b" ]; then
    inference_url="local"
elif [[ $chat_model == *"$subString"* ]]; then
    inference_url="local"
else
    echo "Invalid chat model"
    exit 1
fi

for dataset in "${datasets[@]}";
do
    for split in "${splits[@]}";
    do
        for prompt_method in "${prompt_methods[@]}";
        do
            if [ "cot" = $prompt_method ]; then
                python cot/cot_prompt_inference.py \
                    --dataset $dataset \
                    --split $split \
                    --chat_model $chat_model \
                    --inference_url $inference_url \
                    --prompt_method $prompt_method
            else
                python user_intent/chat_inference.py \
                    --dataset $dataset \
                    --split $split \
                    --chat_model $chat_model \
                    --inference_url $inference_url \
                    --search_engine $search_engine \
                    --prompt_method $prompt_method
            fi
        done
    done
done