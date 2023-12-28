START_TIME=$(date "+%Y%m%d_%H%M%S")

export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_NET_GDR_LEVEL=3
export OMP_NUM_THREADS=8



# 进行训练。注意：
# 1. `streaming` 模式下，要使用 `max_steps` 而不是 `num_train_epochs`
# 2. evaluation_strategy 设为`steps` 开启 do_eval 模式

    

torchrun --nproc_per_node=$HOST_GPU_NUM \
    --nnodes=$HOST_NUM \
    --node_rank=$INDEX \
    --master_addr=$CHIEF_IP \
    --master_port=47963 \
    pretraining/pretraining.py \
    --deepspeed ds_config.json \
    --model_type baichuan \
    --model_name_or_path /dockerdata/Baichuan2-13B-Base/  \
    --output_dir /dockerdata/pt/rank-$INDEX \
    --cache_dir /dockerdata/cache \
    --train_file_dir ./data/pretrain \
    --validation_file_dir ./data/pretrain \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --streaming False \
    --do_train \
    --use_peft False \
    --seed 42 \
    --bf16 \
    --tf32 1 \
    --optim adamw_torch_fused   \
    --lr_scheduler_type cosine \
    --max_grad_norm 0.5 \
    --num_train_epochs 5 \
    --learning_rate 3e-5 \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --logging_strategy steps \
    --logging_steps 5 \
    --eval_steps 50 \
    --evaluation_strategy steps \
    --save_steps 1000 \
    --save_strategy steps \
    --save_total_limit 10 \
    --gradient_accumulation_steps 4 \
    --preprocessing_num_workers `cat /proc/cpuinfo| grep "processor"| wc -l` \
    --block_size 4096 \
    --overwrite_output_dir \
    --ddp_timeout 30000 \
    --logging_first_step True \
    --target_modules W_pack \
    --torch_dtype bfloat16 \
    --device_map auto \
    --report_to tensorboard \
    --ddp_find_unused_parameters False \
    --gradient_checkpointing False \
    --trust_remote_code True
