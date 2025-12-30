#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

MODEL="meta-llama/Llama-2-7b-chat-hf"
ALPHA=0.04
PRUNE_METHOD="wanda_dlp"
SPARSITY_RATIO=0.5
SPARSITY_TYPE="unstructured"
SAVE_MODEL="/acpl-ssd32/k_models/llama2_7b_chat_${PRUNE_METHOD}_${SPARSITY_RATIO}_${SPARSITY_TYPE}_alpha${ALPHA}"

python run.py \
    --model $MODEL \
    --alpha $ALPHA \
    --prune_method $PRUNE_METHOD \
    --sparsity_ratio $SPARSITY_RATIO \
    --sparsity_type $SPARSITY_TYPE \
    --save_model $SAVE_MODEL
