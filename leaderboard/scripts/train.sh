#!/bin/bash

export CARLA_ROOT=${1:-/home/automan-apollo/Dropbox/InterFuser/carla}
export CARLA_SERVER=${CARLA_ROOT}/CarlaUE4.sh
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg

export SRC_ROOT=${2:-/home/automan-apollo/Dropbox/ProFounDrive}

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

python3 ${SRC_ROOT}/corl/train.py 