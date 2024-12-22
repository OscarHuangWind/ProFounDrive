#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 23:26:45 2023

@author: oscar
"""

import os
import random

routes = {}
routes["training_routes/routes_town07_short.xml"] = "scenarios/town07_all_scenarios.json"
routes["training_routes/routes_town07_tiny.xml"] = "scenarios/town07_all_scenarios.json"


ip_ports = []

for port in range(20000, 20090, 10):
    ip_ports.append(("localhost", port, port + 500))


carla_seed = 2000
traffic_seed = 2000

configs = []
for i in range(9):
    configs.append("weather-%d.yaml" % i)


def generate_script(
    ip, port, tm_port, route, scenario, carla_seed, traffic_seed, config_path
):
    lines = []
    lines.append("export HOST=%s\n" % ip)
    lines.append("export PORT=%d\n" % port)
    lines.append("export TM_PORT=%d\n" % tm_port)
    lines.append("export ROUTES=${LEADERBOARD_ROOT}/data/%s\n" % route)
    lines.append("export SCENARIOS=${LEADERBOARD_ROOT}/data/%s\n" % scenario)
    lines.append("export CARLA_SEED=%d\n" % port)
    lines.append("export TRAFFIC_SEED=%d\n" % port)
    lines.append("export TEAM_CONFIG=${YAML_ROOT}/%s\n" % config_path)
    lines.append("export SAVE_PATH=${DATA_ROOT}/%s/data\n" % config_path.split(".")[0])
    lines.append(
        "export CHECKPOINT_ENDPOINT=${DATA_ROOT}/%s/results/%s.json\n"
        % (config_path.split(".")[0], route.split("/")[1].split(".")[0])
    )
    # lines.append(
    #     "export CHECKPOINT_ENDPOINT=${DATA_ROOT}/%s/results/%s.json\n"
    #     % (config_path.split(".")[0], route.split("/")[1].split(".")[0])
    # )
    lines.append("\n")
    base = open("base_script_town07.sh").readlines()

    for line in lines:
        base.insert(17, line)

    return base


for i in range(9):
    if not os.path.exists("bashs"):
        os.mkdir("bashs")
    if not os.path.exists("bashs/weather-%d" % i):
        os.mkdir("bashs/weather-%d" % i)
    for route in routes:
        ip, port, tm_port = ip_ports[i]
        script = generate_script(
            ip,
            port,
            tm_port,
            route,
            routes[route],
            carla_seed,
            traffic_seed,
            configs[i],
        )
        fw = open(
            "bashs/weather-%d/%s.sh" % (i, route.split("/")[1].split(".")[0]), "w"
        )
        for line in script:
            fw.write(line)
