model_path=/home/tanjiejun/python/UserIntent/models/
#model_name=bc2_7B_sft_intent_20231116
model_name=dolly-llama-2-7b-chat-hf-finetune-rewrite
docker_image=bc-training-cluster00-acr-registry.cn-wulanchabu.cr.aliyuncs.com/inferenceservice/baichuan-infer:v0.9.4.15
#docker_image=baichuan-cn-beijing.cr.volces.com/llm/baichuan-infer:v0.9.4.12
port=5567
num_shard=2
#devices=all
devices='"device=6,7"'

docker run -dit --gpus "${devices}" \
        --name ${model_name}-${port} \
        --shm-size 1g -d -p ${port}:80 \
        -v ${model_path}:${model_path} \
        ${docker_image} \
        --model-id ${model_path}/${model_name}  \
        --num-shard 2 \
        --max-total-tokens 4096  \
        --max-input-length 3072  \
        --trust-remote-code