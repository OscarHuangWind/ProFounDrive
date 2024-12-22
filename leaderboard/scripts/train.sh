#!/bin/bash

export SRC_ROOT=${1:-/home/users/ntu/shanhelo/scratch/wenhui_projects/ProFounDrive}
export WS_ROOT=${2:-/home/users/ntu/shanhelo/scratch}

export PYTHONPATH=$PYTHONPATH:${SRC_ROOT}
export PYTHONPATH=$PYTHONPATH:${SRC_ROOT}/corl
export PYTHONPATH=$PYTHONPATH:${SRC_ROOT}/leaderboard
export PYTHONPATH=$PYTHONPATH:${SRC_ROOT}/leaderboard/team_code
export PYTHONPATH=$PYTHONPATH:${SRC_ROOT}/scenario_runner
export LEADERBOARD_ROOT=${SRC_ROOT}/leaderboard
export ROUTES=${SRC_ROOT}/leaderboard/data/42routes/42routes.xml

export PORT=2000 # same as the carla server port #2000
export TM_PORT=8000 # port for traffic manager, required when spawning multiple servers/clients #2500
export DEBUG_CHALLENGE=0
export REPETITIONS=1 # multiple evaluation runs

export TRAIN=True
export BATCH=16
export VLM_FREEZE='prompt_vlm' # MobileVLA
export LLM_FREEZE='llm_model' # MobileVLA & GPTVLA
export ENCODER_FREEZE='vit_encoder' # GPTVLA
export DATA_PATH=${WS_ROOT}/datasets/carlacorl
export SAVE_PATH=${SRC_ROOT}/corl/output/
export OUTPUT_DIR=${SRC_ROOT}/corl/output/

python3 ${SRC_ROOT}/corl/train.py \
--train=${TRAIN} \
--batch_size=${BATCH} \
--vlm_freeze=${VLM_FREEZE} \
--llm_freeze=${LLM_FREEZE} \
--encoder_freeze=${ENCODER_FREEZE} \
--data-path=${DATA_PATH} \
--save-path=${SAVE_PATH} \
--output_dir=${OUTPUT_DIR} \

