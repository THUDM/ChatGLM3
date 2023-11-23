#! /usr/bin/env bash

set -ex

LR=1e-4
NUM_GPUS=4
LORA_RANK=8
MAX_SOURCE_LEN=512
MAX_TARGET_LEN=128
DEV_BATCH_SIZE=4
GRAD_ACCUMULARION_STEPS=2
MAX_STEP=40
SAVE_INTERVAL=20
MAX_SEQ_LEN=512

RUN_NAME=advertise_gen_ft
BASE_MODEL_PATH=THUDM/chatglm3-6b-base
DATASET_PATH=formatted_base_data/advertise_gen.jsonl

DATESTR=`date +%Y%m%d-%H%M%S`
OUTPUT_DIR=output/${RUN_NAME}-${DATESTR}-${LR}
MASTER_PORT=$(shuf -n 1 -i 10000-65535)

mkdir -p $OUTPUT_DIR

torchrun --standalone --nnodes=1 --nproc_per_node=$NUM_GPUS finetune.py \
    --train_format input-output \
    --train_file $DATASET_PATH \
    --lora_rank $LORA_RANK \
    --max_seq_length $MAX_SEQ_LEN \
    --preprocessing_num_workers 1 \
    --model_name_or_path $BASE_MODEL_PATH \
    --output_dir $OUTPUT_DIR \
    --per_device_train_batch_size $DEV_BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUMULARION_STEPS \
    --max_steps $MAX_STEP \
    --logging_steps 1 \
    --save_steps $SAVE_INTERVAL \
    --learning_rate $LR  2>&1 | tee ${OUTPUT_DIR}/train.log
