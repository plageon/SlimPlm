#!/bin/bash
export dataset="v0129"
export base_model="TinyLlama-1.1B-Chat"
#export base_model="llama-2-7b-chat-hf"
export method="query-rewrite"
#export method="search-tag"
export task_name="${dataset}-${base_model}-${method}"
export WANDB_DISABLED=True
export eval_steps=20

deepspeed --include="localhost:4,5,6,7" user_intent/finetune_llm.py \
--task_name $task_name \
--do_train \
--do_eval \
--model_name_or_path "../../huggingface/${base_model}" \
--output_dir "models/${task_name}" \
--data_path "./user_intent_data/mixed/${dataset}" \
--dataset $dataset \
--per_device_train_batch_size 8 \
--per_device_eval_batch_size 8 \
--learning_rate 1e-4 \
--num_train_epochs 2 \
--weight_decay 0.01 \
--save_total_limit 2 \
--load_best_model_at_end  \
--greater_is_better False \
--evaluation_strategy "steps" \
--save_strategy "steps" \
--logging_strategy "steps" \
--eval_steps $eval_steps \
--logging_steps $eval_steps \
--save_steps $eval_steps \
--metric_for_best_model "loss" \
--warmup_ratio 0.1 \
--gradient_accumulation_steps 2 \
--eval_accumulation_steps 8 \
--seed 1234 \
--predict_task \
--predict_search_tags \
--predict_questions \
--deepspeed ./ds_configs/zero_stage2.json

#--evaluation_strategy "steps" \
#--save_strategy "steps" \
#--logging_strategy "steps" \
#--eval_steps $eval_steps \
#--logging_steps $eval_steps \
#--save_steps $eval_steps \

#--evaluation_strategy "epoch" \
#--save_strategy "epoch" \
#--logging_strategy "epoch" \

#--resume_from_checkpoint "./models/v1221-llama-2-7b-chat-hf-finetune-rewrite/checkpoint-504" \

#--provide_without_search_answer \
#--predict_task \
#--predict_search_tags \
#--predict_questions \
#--predict_claims \