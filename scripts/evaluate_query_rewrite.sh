#!/bin/bash
export dataset="dolly"
export base_model="llama-2-7b-chat-hf"
export method="finetune-rewrite"
export task_name="${dataset}-${base_model}-${method}"
export WANDB_DISABLED=True
export eval_steps=200

deepspeed --include="localhost:0,1,2,3,4,5,6,7" user_intent/finetune_llm.py \
--task_name $task_name \
--do_eval  \
--model_name_or_path "../../huggingface/${base_model}" \
--output_dir "models/${task_name}" \
--data_path "user_intent_data/dolly/gpt4/" \
--dataset $dataset \
--per_device_train_batch_size 2 \
--per_device_eval_batch_size 2 \
--learning_rate 1e-4 \
--num_train_epochs 1 \
--weight_decay 0.01 \
--save_total_limit 1 \
--load_best_model_at_end  \
--greater_is_better True \
--evaluation_strategy "steps" \
--save_strategy "steps" \
--logging_strategy "steps" \
--eval_steps $eval_steps \
--logging_steps $eval_steps \
--save_steps $eval_steps \
--metric_for_best_model "rougeL" \
--warmup_ratio 0.1 \
--gradient_accumulation_steps 2 \
--eval_accumulation_steps 8 \
--seed 1234 \
--deepspeed ./ds_configs/zero_stage2.json
