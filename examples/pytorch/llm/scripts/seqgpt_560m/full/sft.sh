CUDA_VISIBLE_DEVICES=0 \
python src/llm_sft.py \
    --model_type seqgpt-560m \
    --sft_type full \
    --template_type default-generation \
    --dtype bf16 \
    --output_dir runs \
    --dataset ner-jave-zh \
    --dataset_sample -1 \
    --num_train_epochs 3 \
    --max_length 1024 \
    --gradient_checkpointing false \
    --batch_size 32 \
    --weight_decay 0.01 \
    --learning_rate 2e-5 \
    --gradient_accumulation_steps 1 \
    --max_grad_norm 1 \
    --warmup_ratio 0.03 \
    --eval_steps 100 \
    --save_steps 100 \
    --only_save_model false \
    --save_total_limit 2 \
    --logging_steps 10 \
    --push_to_hub false \
    --hub_model_id seqgpt-560m-full \
    --hub_private_repo true \
    --hub_token 'your-sdk-token' \
