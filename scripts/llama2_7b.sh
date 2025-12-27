#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

MODEL="meta-llama/Llama-2-7b-hf"
ALPHA=0.15
PRUNE_METHOD="wanda_dlp"
SPARSITY_RATIO=0.7
SPARSITY_TYPE="unstructured"

python   run.py \
    --model $MODEL \
    --alpha $ALPHA \
    --prune_method $PRUNE_METHOD \
    --sparsity_ratio $SPARSITY_RATIO \
    --sparsity_type $SPARSITY_TYPE \

# sparsity 0.7 unstructured with alpha 0.15
python   run.py \
    --model "meta-llama/Llama-2-7b-hf" \
    --alpha 0.15 \
    --prune_method "wanda_dlp" \
    --sparsity_ratio 0.7 \
    --sparsity_type "unstructured" \
    --save_model "outputs/llama2_7b_wanda_dlp_0.7_unstructured_alpha0.15"

# sparsity 0.6 unstructured with alpha 0.1
python run.py \
    --model "meta-llama/Llama-2-7b-hf" \
    --alpha 0.1 \
    --prune_method "wanda_dlp" \
    --sparsity_ratio 0.6 \
    --sparsity_type "unstructured" \
    --save_model "outputs/llama2_7b_wanda_dlp_0.6_unstructured_alpha0.1"

# sparsity 0.5 unstructured with alpha 0.04
python run.py \
    --model "meta-llama/Llama-2-7b-hf" \
    --alpha 0.04 \
    --prune_method "wanda_dlp" \
    --sparsity_ratio 0.5 \
    --sparsity_type "unstructured" \
    --save_model "outputs/llama2_7b_wanda_dlp_0.5_unstructured_alpha0.04"
