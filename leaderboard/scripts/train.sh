#!/bin/bash

#!/bin/bash

# export SRC_ROOT="${1:-/root/oscar_projects/ProFounDrive}"
# export WS_ROOT="${2:-/root/oscar_projects}"

export SRC_ROOT="${1:-/home/automan-apollo/Dropbox/ProFounDrive}"
export WS_ROOT="${2:-/home/automan-apollo/Dropbox/VisionFoundationVehicle/corl}"

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

MODE="eval"
BATCH=32
EPOCHS="50 30 30" #"70 50 50"
DATASET="min-3-domain"
LEARNING_MODE="Split-sequential"
PRETRAIN="pretrain"
VLM_FREEZE="prompt_vlm" # for MobileVLM-based; only freeze after 1st task
LLM_FREEZE="llm_model" # for GPT-based; only freeze after 1st task
ENCODER_FREEZE="vit_encoder" # for GPT-based; only freeze after 1st task
DECODER_FREEZE="prompt_gpt" # for GPT-based; only freeze after 1st task
DATA_PATH="/home/automan-apollo/profoundrive_tmp/CarlaCORL"
SAVE_PATH="${WS_ROOT}/output"
OUTPUT_DIR="${WS_ROOT}/output"
WANDBKEY="8726d9823ea1bc5190c32369f254d61eab02a17b"

python3 "${SRC_ROOT}/corl/train.py" \
  --mode="${MODE}" \
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
  --output-dir="${OUTPUT_DIR}"\
  --wandb="${WANDBKEY}"

