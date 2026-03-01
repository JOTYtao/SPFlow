#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="python"
TRAIN_PY="/home/joty/code/FM_interpolation/train.py"
RNN_PATH="/home/joty/code/FM_interpolation/checkpoints/SPTransformer/epoch=72-val_loss=0.057832.ckpt"
NUM_GPUS=1   
MAX_EPOCHS=1000
PRECISION="bf16"
FLOW_STEPS=50
CHECK_VAL_EVERY=1
EMA_DECAY=0.9999
EPS=0.0
CKPT_DIR="/home/joty/code/FM_interpolation/checkpoints/spflow"  
RESUME_CKPT=""
WEIGHT_DECAY=1e-2
LR=1.e-4
# W&B
WANDB_PROJECT="flow_matching"
WANDB_GROUP="WIPFlowNWP"
WANDB_ID=""           

# ====================================

export WANDB_MODE=online
export WANDB_DIR="${HOME}/wandb_logs"
export CUDA_VISIBLE_DEVICES=1
${PYTHON_BIN} "${TRAIN_PY}" \
  --precision "${PRECISION}" \
  --num_flow_steps ${FLOW_STEPS} \
  --num_gpus ${NUM_GPUS} \
  --rnn_checkpoint_path ${RNN_PATH} \
  --check_val_every_n_epoch ${CHECK_VAL_EVERY} \
  --checkpoint_path ${CKPT_DIR}\
  --max_epochs ${MAX_EPOCHS} \
  --ema_decay ${EMA_DECAY} \
  --eps ${EPS} \
  --weight_decay ${WEIGHT_DECAY} \
  --lr ${LR} \
  --wandb_project_name "${WANDB_PROJECT}" \
  --wandb_group "${WANDB_GROUP}" \
  --resume_from_ckpt "${RESUME_CKPT}" 
  $( [ -n "${WANDB_ID}" ] && echo --wandb_id "${WANDB_ID}" ) \
