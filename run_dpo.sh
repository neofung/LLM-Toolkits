#START_TIME=$(date "+%Y%m%d_%H%M%S")
START_TIME=$(date "+%Y%m%d-%H")

export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_NET_GDR_LEVEL=3
export OMP_NUM_THREADS=8

per_device_train_batch_size=4
CUDA_VISIBLE_DEVICES=0,1,2,3 python \
    dpo/dpo_training.py \
    --model_name_or_path /dockerdata/Baichuan2-13B-Chat/  \
    --model_type baichuan \
    --use_flash_attention_2 False \
    --cache_dir /dockerdata/cache \
    --train_file_dir ./data/reward/ \
    --validation_split_percentage 10 \
    --output_dir /dockerdata/dpo_Baichuan2-13B-Chat_${START_TIME}/rank-$INDEX \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --per_device_eval_batch_size 2 \
    --max_source_length 4096 \
    --max_target_length 2048 \
    --do_train \
    --do_eval \
    --use_peft False \
    --bf16 \
    --optim adamw_torch   \
    --lr_scheduler_type cosine \
    --warmup_steps 0 \
    --learning_rate 2e-6 \
    --weight_decay 0. \
    --num_train_epochs 5 \
    --logging_steps 2 \
    --eval_steps 10 \
    --evaluation_strategy steps \
    --save_strategy epoch \
    --save_total_limit 0 \
    --gradient_accumulation_steps 1 \
    --preprocessing_num_workers `cat /proc/cpuinfo| grep "processor"| wc -l` \
    --torch_dtype bfloat16 \
    --fp16 False \
    --device_map auto \
    --report_to tensorboard \
    --gradient_checkpointing False \
    --trust_remote_code True
