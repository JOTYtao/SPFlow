#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="python"
INFER_PY="/home/joty/code/FM_interpolation/inference.py"
CKPT_PATH="/home/joty/code/FM_interpolation/checkpoints/spflow3/epoch=59-val_loss=0.070746.ckpt"
RNN_PATH="/home/joty/code/FM_interpolation/checkpoints/SPTransformer/epoch=72-val_loss=0.057832.ckpt"
CONFIG_PATH="/home/joty/code/FM_interpolation/config/config.yaml"
OUT_DIR="/home/joty/code/FM_interpolation/results/spflow/"

NUM_GPUS=1
PRECISION="32"
MAX_SAMPLES=-1   
USE_TEST_SPLIT=1  

LR=5e-4
WEIGHT_DECAY=1e-3
BETAS0=0.9
BETAS1=0.95
NUM_FLOW_STEPS=50
EMA_DECAY=0.9999
EPS=0.0


ENABLE_WANDB=0
WANDB_PROJECT="flow_matching"
WANDB_GROUP="inference"
WANDB_ID=""

if [[ ! -f "${CKPT_PATH}" ]]; then
  echo "Checkpoint not found: ${CKPT_PATH}"
  exit 1
fi

CMD=(
  "${PYTHON_BIN}" "${INFER_PY}"
  --rnn_checkpoint_path "${RNN_PATH}"
  --ckpt_path "${CKPT_PATH}"
  --config_path "${CONFIG_PATH}"
  --output_dir "${OUT_DIR}"
  --num_gpus ${NUM_GPUS}
  --precision "${PRECISION}"
  --lr ${LR}
  --weight_decay ${WEIGHT_DECAY}
  --betas ${BETAS0} ${BETAS1}
  --num_flow_steps ${NUM_FLOW_STEPS}
  --ema_decay ${EMA_DECAY}
  --eps ${EPS}
)

if [[ "${USE_TEST_SPLIT}" -eq 1 ]]; then
  CMD+=( --use_test_split )
fi

if [[ "${ENABLE_WANDB}" -eq 1 ]]; then
  export WANDB_MODE=online
  export WANDB_DIR="${HOME}/wandb_logs"
  CMD+=( --enable_wandb --wandb_project_name "${WANDB_PROJECT}" --wandb_group "${WANDB_GROUP}" )
  if [[ -n "${WANDB_ID}" ]]; then
    CMD+=( --wandb_id "${WANDB_ID}" )
  fi
else
  export WANDB_MODE=disabled
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"