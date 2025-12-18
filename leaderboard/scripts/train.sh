#!/bin/bash

#!/bin/bash

export SRC_ROOT="${1:-/root/oscar_projects/ProFounDrive}"
export WS_ROOT="${2:-/root/oscar_projects}"

export PYTHONPATH="$PYTHONPATH:${SRC_ROOT}"
export PYTHONPATH="$PYTHONPATH:${SRC_ROOT}/corl"
export PYTHONPATH="$PYTHONPATH:${SRC_ROOT}/leaderboard"
export PYTHONPATH="$PYTHONPATH:${SRC_ROOT}/leaderboard/team_code"
export PYTHONPATH="$PYTHONPATH:${SRC_ROOT}/scenario_runner"

export LEADERBOARD_ROOT="${SRC_ROOT}/leaderboard"
export ROUTES="${SRC_ROOT}/leaderboard/data/42routes/42routes.xml"

export PORT=2000
export TM_PORT=8000
export DEBUG_CHALLENGE=0
export REPETITIONS=1

mode="train" #"eval"
BATCH=32
EPOCHS="70 50 50"
DATASET="min-3-domain"
LEARNING_MODE="Split-sequential"
PRETRAIN="scratch" # "pretrain"
VLM_FREEZE="prompt_vlm" # for MobileVLM-based; only freeze after 1st task
LLM_FREEZE="llm_model" # for GPT-based; only freeze after 1st task
ENCODER_FREEZE="vit_encoder" # for GPT-based; only freeze after 1st task
DECODER_FREEZE="prompt_gpt" # for GPT-based; only freeze after 1st task
DATA_PATH="${WS_ROOT}/data/carlacorl"
SAVE_PATH="${WS_ROOT}/output"
OUTPUT_DIR="${WS_ROOT}/output"

python3 "${SRC_ROOT}/corl/train.py" \
  --mode="${mode}" \
  --batch_size="${BATCH}" \
  --epochs ${EPOCHS} \
  --setting="${DATASET}" \
  --dataset_mode="${LEARNING_MODE}" \
  --load-pretrain="${PRETRAIN}" \
  --vlm_freeze="${VLM_FREEZE}" \
  --llm_freeze="${LLM_FREEZE}" \
  --encoder_freeze="${ENCODER_FREEZE}" \
  --decoder_freeze="${DECODER_FREEZE}" \
  --data-path="${DATA_PATH}" \
  --save-path="${SAVE_PATH}" \
  --output-dir="${OUTPUT_DIR}"

# export SRC_ROOT=${1:-/root/oscar_projects/ProFounDrive}
# export WS_ROOT=${2:-/root/oscar_projects}

# export PYTHONPATH=$PYTHONPATH:${SRC_ROOT}
# export PYTHONPATH=$PYTHONPATH:${SRC_ROOT}/corl
# export PYTHONPATH=$PYTHONPATH:${SRC_ROOT}/leaderboard
# export PYTHONPATH=$PYTHONPATH:${SRC_ROOT}/leaderboard/team_code
# export PYTHONPATH=$PYTHONPATH:${SRC_ROOT}/scenario_runner
# export LEADERBOARD_ROOT=${SRC_ROOT}/leaderboard
# export ROUTES=${SRC_ROOT}/leaderboard/data/42routes/42routes.xml

# export PORT=2000 # same as the carla server port #2000
# export TM_PORT=8000 # port for traffic manager, required when spawning multiple servers/clients #2500
# export DEBUG_CHALLENGE=0
# export REPETITIONS=1 # multiple evaluation runs

# mode='eval'
# BATCH=16
# VLM_FREEZE='prompt_vlm' # MobileVLA
# LLM_FREEZE='llm_model' # MobileVLA & GPTVLA
# ENCODER_FREEZE='vit_encoder' # GPTVLA
# DATA_PATH=${WS_ROOT}/data/carlacorl
# SAVE_PATH=${SRC_ROOT}/output/
# OUTPUT_DIR=${SRC_ROOT}/output/

# python3 ${SRC_ROOT}/corl/train.py \
# --mode=${mode} \
# --batch_size=${BATCH} \
# --vlm_freeze=${VLM_FREEZE} \
# --llm_freeze=${LLM_FREEZE} \
# --encoder_freeze=${ENCODER_FREEZE} \
# --data-path=${DATA_PATH} \
# --save-path=${SAVE_PATH} \
# --output-dir=${OUTPUT_DIR}

