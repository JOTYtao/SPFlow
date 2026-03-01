#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="python"
TRAIN_PY="/home/joty/code/FM_interpolation/train_RNN.py"
OUT_DIR="/home/joty/code/FM_interpolation/results/LSTM"
CKPT_PATH="/home/joty/code/FM_interpolation/checkpoints/SPTransformer1/epoch=12-val_loss=0.065134.ckpt"
NUM_GPUS=1   
MAX_EPOCHS=1000
PRECISION="bf16"
FLOW_STEPS=100
CHECK_VAL_EVERY=1
EMA_DECAY=0.9999
EPS=0.0
CKPT_DIR="/home/joty/code/FM_interpolation/checkpoints/SPTransformer"  
RESUME_CKPT=""
WEIGHT_DECAY=1e-2
LR=1.e-3
# W&B
WANDB_PROJECT="RNN"
WANDB_GROUP="RNN"
WANDB_ID=""           

# ====================================

export WANDB_MODE=online
export WANDB_DIR="${HOME}/wandb_logs"
export CUDA_VISIBLE_DEVICES=1
${PYTHON_BIN} "${TRAIN_PY}" \
  --precision "${PRECISION}" \
  --num_gpus ${NUM_GPUS} \
  --check_val_every_n_epoch ${CHECK_VAL_EVERY} \
  --checkpoint_path ${CKPT_DIR}\
  --ckpt_path ${CKPT_PATH}\
  --max_epochs ${MAX_EPOCHS} \
  --lr ${LR} \
  --output_dir "${OUT_DIR}" \
  --wandb_project_name "${WANDB_PROJECT}" \
  --wandb_group "${WANDB_GROUP}" \
  --resume_from_ckpt "${RESUME_CKPT}" 
  $( [ -n "${WANDB_ID}" ] && echo --wandb_id "${WANDB_ID}" ) \
