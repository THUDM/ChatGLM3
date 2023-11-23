#! /usr/bin/env bash

set -ex

LR=1e-4
NUM_GPUS=4
MAX_SEQ_LEN=2048
DEV_BATCH_SIZE=16
GRAD_ACCUMULARION_STEPS=1
MAX_STEP=200
SAVE_INTERVAL=50

DATESTR=`date +%Y%m%d-%H%M%S`
RUN_NAME=tool_alpaca_ft
DATASET_PATH=formatted_data/tool_alpaca.jsonl

BASE_MODEL_PATH=THUDM/chatglm3-6b
OUTPUT_DIR=output/${RUN_NAME}-${DATESTR}-${LR}

mkdir -p $OUTPUT_DIR

torchrun --standalone --nnodes=1 --nproc_per_node=$NUM_GPUS finetune.py \
    --train_format multi-turn \
    --train_file $DATASET_PATH \
    --max_seq_length $MAX_SEQ_LEN \
    --preprocessing_num_workers 1 \
    --model_name_or_path $BASE_MODEL_PATH \
    --output_dir $OUTPUT_DIR \
    --per_device_train_batch_size $DEV_BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUMULARION_STEPS \
    --max_steps $MAX_STEP \
    --logging_steps 1 \
    --save_steps $SAVE_INTERVAL \
    --fp16 \
    --deepspeed configs/deepspeed.json 2>&1 | tee ${OUTPUT_DIR}/train.log
