#START_TIME=$(date "+%Y%m%d_%H%M%S")
START_TIME=$(date "+%Y%m%d-%H")

export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_NET_GDR_LEVEL=3
export OMP_NUM_THREADS=8


per_device_train_batch_size=1
torchrun --nproc_per_node=$HOST_GPU_NUM \
    --nnodes=$HOST_NUM \
    --node_rank=$INDEX \
    --master_addr=$CHIEF_IP \
    --master_port=47963 \
    supervised_finetuning/supervised_finetuning.py \
    --deepspeed ds_config.json \
    --model_type baichuan \
    --template_name baichuan2 \
    --model_name_or_path /dockerdata/Baichuan2-13B-Chat  \
    --cache_dir /dockerdata/cache \
    --train_file_dir ./data/supervised_fine_tuning/alpaca-gpt4-data/data/ \
    --validation_split_percentage 5 \
    --drop_out_of_length_limit_sample True \
    --output_dir /dockerdata/sft_baichuan2-13b_${START_TIME}/rank-$INDEX \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --per_device_eval_batch_size 1 \
    --max_source_length 4096 \
    --max_target_length 512 \
    --do_train \
    --use_peft False \
    --bf16 \
    --tf32 1 \
    --optim adamw_torch_fused   \
    --lr_scheduler_type cosine \
    --max_grad_norm 0.5 \
    --num_train_epochs 2 \
    --learning_rate 1e-5 \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --logging_strategy steps \
    --logging_steps 5 \
    --eval_steps 20 \
    --evaluation_strategy steps \
    --save_steps 100 \
    --save_strategy epoch \
    --save_total_limit 10 \
    --gradient_accumulation_steps `expr 64 / ${per_device_train_batch_size} / ${HOST_NUM} / ${HOST_GPU_NUM}` \
    --preprocessing_num_workers `cat /proc/cpuinfo| grep "processor"| wc -l` \
    --overwrite_output_dir \
    --ddp_timeout 30000 \
    --logging_first_step True \
    --torch_dtype bfloat16 \
    --device_map auto \
    --report_to tensorboard \
    --ddp_find_unused_parameters False \
    --gradient_checkpointing False \
    --trust_remote_code True
