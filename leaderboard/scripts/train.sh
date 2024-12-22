#!/bin/bash

#export CARLA_ROOT=${1:-/home/spyder/project/LLMs/carla}
export CARLA_ROOT=${1:-/home/automan-apollo/projects/zsy_projects/MobileVLM_Drive/carla}
export CARLA_SERVER=${CARLA_ROOT}/CarlaUE4.sh
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg

# export SRC_ROOT=${2:-/home/automan-apollo/Dropbox/VisionFoundationVehicle}
#export SRC_ROOT=${2:-/home/users/ntu/wenhui00/scratch/projects/VLMDrive-Pro}
#export SRC_ROOT=${2:-/home/users/ntu/songyan0/scratch/projects/VLMDrive-Pro}
#export SRC_ROOT=${2:-/home/spyder/Dropbox/VisionFoundationVehicle}
export SRC_ROOT=${2:-/home/users/ntu/shanhelo/scratch/wenhui_projects/VLMDrive-Pro}

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

#export TEAM_AGENT=${SRC_ROOT}/corl/carla_evaluate/corl_agent.py # agent
#export TEAM_CONFIG=${SRC_ROOT}/corl/corl_config.py
#export CHECKPOINT_ENDPOINT=${SRC_ROOT}/results/corl_result.json # results file
#export SCENARIOS=${SRC_ROOT}/leaderboard/data/scenarios/town07_all_scenarios.json
#export ROUTES=${SRC_ROOT}/leaderboard/data/42routes/42routes.xml #validation_routes/routes_town07_sample.xml
#export SAVE_PATH=data/eval # path for saving episodes while evaluating
#export RESUME=False #True

python3 ${SRC_ROOT}/corl/train.py 
# CUDA_VISIBLE_DEVICE=0 python3 ${SRC_ROOT}/corl/train.py 