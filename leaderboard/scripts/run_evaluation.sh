#!/bin/bash

#export CARLA_ROOT=carla
export CARLA_ROOT=${1:-/home/automan-apollo/Dropbox/InterFuser/carla}
export CARLA_SERVER=${CARLA_ROOT}/CarlaUE4.sh
export PYTHONPATH=${CARLA_ROOT}/PythonAPI:$PYTHONPATH
export PYTHONPATH=${CARLA_ROOT}/PythonAPI/carla:$PYTHONPATH
export PYTHONPATH=${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg:$PYTHONPATH

export SRC_ROOT=${2:-/home/automan-apollo/Dropbox/ProFounDrive}
export PYTHONPATH=$PYTHONPATH:${SRC_ROOT}
export PYTHONPATH=$PYTHONPATH:${SRC_ROOT}/corl
export PYTHONPATH=${SRC_ROOT}/leaderboard:$PYTHONPATH
export PYTHONPATH=$PYTHONPATH:${SRC_ROOT}/leaderboard/team_code
export PYTHONPATH=$PYTHONPATH:${SRC_ROOT}/scenario_runner
export LEADERBOARD_ROOT=${SRC_ROOT}/leaderboard

export CHALLENGE_TRACK_CODENAME=SENSORS
export PORT=2000 # same as the carla server port #2000
export TM_PORT=8000 # port for traffic manager, required when spawning multiple servers/clients
export DEBUG_CHALLENGE=0
export REPETITIONS=1 # multiple evaluation runs

export TEAM_AGENT=${SRC_ROOT}/corl/carla_evaluate/corl_agent.py # agent
export TEAM_CONFIG=${SRC_ROOT}/corl/corl_config.py
export CHECKPOINT_ENDPOINT=${SRC_ROOT}/results/corl_result_highway_longtail.json # results file
export SCENARIOS=${SRC_ROOT}/leaderboard/data/scenarios/town03_all_scenarios.json
#export ROUTES=${SRC_ROOT}/leaderboard/data/42routes/42routes.xml #validation_routes/routes_town04_sample.xml
#export ROUTES=${SRC_ROOT}/leaderboard/data/validation_routes/routes_town03_longtail.xml
export ROUTES=${SRC_ROOT}/leaderboard/data/longtail_routes/routes_town04_longtail.xml
export SAVE_PATH=${SRC_ROOT}/eval_data # path for saving episodes while evaluating
export RESUME=False #True

# python3 ${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator.py \
python3 ${SRC_ROOT}/corl/evaluation.py \
--scenarios=${SCENARIOS}  \
--routes=${ROUTES} \
--repetitions=${REPETITIONS} \
--track=${CHALLENGE_TRACK_CODENAME} \
--checkpoint=${CHECKPOINT_ENDPOINT} \
--agent=${TEAM_AGENT} \
--agent-config=${TEAM_CONFIG} \
--debug=${DEBUG_CHALLENGE} \
--resume=${RESUME} \
--port=${PORT} \
--trafficManagerPort=${TM_PORT}

