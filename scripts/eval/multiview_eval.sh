#!/bin/bash
# Multi-View Temporal Ensembling Evaluation Script
# This script runs inference with multiple temporal views and ensembles the results.

# ========== Configuration ==========
NUM_VIEWS=5  # Number of temporal views (adjust as needed)
DATASET="nextqa"  # Options: nextqa, egos
SPLIT="val"  # Options: val, test
MODEL="7B"
CHECKPOINT="vqa_checkpoint/checkpoint_pretrain/llama2_7b_acc4_br5e3_correct_vnips/checkpoint_19.pth"

# ========== Run Multi-View Inference ==========
python multiview_inference.py \
    --model ${MODEL} \
    --dataset ${DATASET} \
    --split ${SPLIT} \
    --num_views ${NUM_VIEWS} \
    --max_seq_len 128 \
    --batch_size 10 \
    --bias 3 \
    --tau 100. \
    --max_feats 10 \
    --resume ${CHECKPOINT} \
    --adapter_len 50 \
    --output_dir results/multiview/${DATASET}_${NUM_VIEWS}views \
    --llama2 \
    --llama_model_path ./pretrained/llama2/ \
    --memory

echo "=========================================="
echo "Multi-View Inference Complete!"
echo "Results saved to: results/multiview/${DATASET}_${NUM_VIEWS}views"
echo "=========================================="
