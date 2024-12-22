#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 25 02:29:31 2023

@author: oscar
"""

import os

routes = {}

routes["training_routes/routes_town06_short.xml"] = "scenarios/town06_all_scenarios.json"
routes["training_routes/routes_town06_tiny.xml"] = "scenarios/town06_all_scenarios.json"



routes_list = []
for route in routes:
    routes_list.append(route.split("/")[1].split(".")[0])

if not os.path.exists("batch_run"):
    os.mkdir("batch_run")

for route in routes_list:
    fw = open("batch_run/run_route_%s.sh" % route, "w")
    for i in range(9):
        fw.write("bash data_collection/bashs/weather-%d/%s.sh & \n" % (i, route))
