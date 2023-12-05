#! /usr/bin/env bash

set -ex

PRE_SEQ_LEN=512
LR=2e-2
NUM_GPUS=1
MAX_SEQ_LEN=2048
DEV_BATCH_SIZE=1
GRAD_ACCUMULARION_STEPS=16
MAX_STEP=1000
SAVE_INTERVAL=500

DATESTR=`date +%Y%m%d-%H%M%S`
RUN_NAME=tool_alpaca_pt

BASE_MODEL_PATH=E:\\chatglm3-6b
DATASET_PATH=D:\\ChatGLM3\\finetune_chatmodel_demo\\train_help.json
OUTPUT_DIR=output/${RUN_NAME}-${DATESTR}-${PRE_SEQ_LEN}-${LR}

mkdir -p $OUTPUT_DIR

python finetune.py \
    --train_format input-output \
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
    --learning_rate $LR \
    --pre_seq_len $PRE_SEQ_LEN 2>&1 | tee ${OUTPUT_DIR}/train.log
