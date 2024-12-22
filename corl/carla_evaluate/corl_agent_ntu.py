#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 22:11:29 2024

@author: oscar
"""

import sys
# sys.path.append('home/oscar/Dropbox/InterFuser')
# sys.path.append('home/automan/Dropbox/InterFuser')

import os
import json
import time
import datetime
import pathlib
import time
import imp
import cv2
from collections import deque
import matplotlib.pyplot as plt

import torch
import numpy as np
from PIL import Image
from easydict import EasyDict

import carla
from agents.navigation.local_planner import RoadOption
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from carla_birdeye_view import BirdViewProducer, BirdViewCropType, PixelDimensions

from torchvision import transforms
from leaderboard.autoagents import autonomous_agent
from timm.models import create_model
from team_code.map_agent import MapAgent
from team_code.utils import lidar_to_histogram_features, transform_2d_points
from team_code.planner import RoutePlanner
from team_code.render import render, render_self_car, render_waypoints
from team_code.tracker import Tracker

from corl_config import GlobalConfig
from carla_evaluate.corl_controller import PIDController, CORLController
from foundation_model.gpt_vla_agent import init_gpt_vla
from foundation_model.llama_vla_agent import init_llama_vla
from dataset_preprocess.data import CARLA_Data

import math
import yaml

try:
    import pygame
except ImportError:
    raise RuntimeError("cannot import pygame, make sure pygame package is installed")


SAVE_PATH = os.environ.get("SAVE_PATH", 'eval')
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

WEATHERS = {
    "ClearNoon": carla.WeatherParameters(5.0, 0.0, 0.0, 0.35, 45.0, 75.0, 0.0, 0.0, 0.0, 0.0),
    "ClearSunset": carla.WeatherParameters(5.0, 0.0, 0.0, 0.35, 45.0, 5.0, 0.0, 0.0, 0.0, 0.0),
    "ClearNight": carla.WeatherParameters(5.0, 0.0, 0.0, 0.35, -1.0, -90.0, 0.0, 0.0, 0.0, 0.0),
    # "ClearNight": carla.WeatherParameters(5.0, 0.0, 0.0, 0.35, -1.0, -90.0, 60.0, 75.0, 1.0, 0.0),
    # "CloudyMorning": carla.WeatherParameters(60.0, 0.0, 30.0, 10.0, 45.0, 35.0, 0.0, 0.0, 0.0, 0.0),
    # "CloudyDawn": carla.WeatherParameters(60.0, 0.0, 30.0, 10.0, 45.0, 5.0, 0.0, 0.0, 0.0, 0.0),
    "FogyNoon": carla.WeatherParameters(60.0, 0.0, 50.0, 10.0, 45.0, 75.0, 40.0, 0.75, 1.0, 0.0),
    "FogyySunset": carla.WeatherParameters(60.0, 0.0, 50.0, 10.0, 45.0, 5.0, 20.0, 0.75, 1.0, 0.0),
    "FogyNight": carla.WeatherParameters(60.0, 0.0, 50.0, 10.0, -1.0, -90.0, 60.0, 0.75, 1.0, 0.0),
    "HardRainNoon": carla.WeatherParameters(100.0, 100.0, 70.0, 50.0, 45.0, 75.0, 0.0, 0.0, 0.0, 100.0),
    "HardRainSunset": carla.WeatherParameters(100.0, 100.0, 70.0, 50.0, 45.0, 5.0, 0.0, 0.0, 0.0, 100.0),
    # "HardRainNight": carla.WeatherParameters(100.0, 100.0, 70.0, 50.0, -1.0, -90.0, 100.0, 0.75, 0.1, 100.0),
    "HardRainNight": carla.WeatherParameters(100.0, 100.0, 70.0, 50.0, -1.0, -90.0, 60.0, 30.0, 1.0, 100.0),
    # "ClearNoon": carla.WeatherParameters.ClearNoon,
    # "ClearSunset": carla.WeatherParameters.ClearSunset,
    #"CloudyNoon": carla.WeatherParameters.CloudyNoon,
    #"CloudySunset": carla.WeatherParameters.CloudySunset,
    #"WetNoon": carla.WeatherParameters.WetNoon,
    #"WetSunset": carla.WeatherParameters.WetSunset,
    #"WetNight": carla.WeatherParameters(5.0,0.0,50.0,10.0,-1.0,-90.0,60.0,75.0,1.0,60.0),
    #"WetCloudyNoon": carla.WeatherParameters.WetCloudyNoon,
    #"WetCloudySunset": carla.WeatherParameters.WetCloudySunset,
    #"WetCloudyNight": carla.WeatherParameters(60.0,0.0,50.0,10.0,-1.0,-90.0,60.0,0.75,0.1,60.0),
    #"SoftRainNoon": carla.WeatherParameters.SoftRainNoon,
    #"SoftRainSunset": carla.WeatherParameters.SoftRainSunset,
    #"SoftRainNight": carla.WeatherParameters(60.0,30.0,50.0,30.0,-1.0,-90.0,60.0,0.75,0.1,60.0),
    # "MidRainyNoon": carla.WeatherParameters.MidRainyNoon,
    # "MidRainSunset": carla.WeatherParameters.MidRainSunset,
    # "MidRainyNight": carla.WeatherParameters(80.0,60.0,60.0,60.0,-1.0,-90.0,60.0,0.75,0.1,80.0),
    # "HardRainNoon": carla.WeatherParameters.HardRainNoon,
    # "HardRainSunset": carla.WeatherParameters.HardRainSunset,
}
WEATHERS_IDS = list(WEATHERS)

LONG_DECISION_LIST = ['Acc', 'Slow Down', 'Brake']
LAT_DECISION_LIST = ['Left', 'Right', 'STRAIGHT', 'Keep Lane', 'ChangeLane Left', 'ChangeLane Right']

class DisplayInterface(object):
    def __init__(self):
        self._width = 1200
        self._height = 600
        self._surface = None

        pygame.init()
        pygame.font.init()
        self._clock = pygame.time.Clock()
        self._display = pygame.display.set_mode(
            (self._width, self._height), pygame.HWSURFACE | pygame.DOUBLEBUF
        )

        # self._vehicle = CarlaDataProvider.get_hero_actor()
        # self._world = self._vehicle.get_world()
        # self._map = self._world.get_map()
        # self.birdview_producer = BirdViewProducer(
        #     CarlaDataProvider.get_client(),  # carla.Client
        #     target_size=PixelDimensions(width=400, height=400),
        #     pixels_per_meter=4,
        #     crop_type=BirdViewCropType.FRONT_AND_REAR_AREA,
        # )

        pygame.display.set_caption("CORL Agent")

    def run_interface(self, input_data):
        rgb = input_data['rgb']
        rgb_left = input_data['rgb_left']
        rgb_right = input_data['rgb_right']
        rgb_focus = input_data['rgb_focus']
        # map = input_data['map']
        trajectory = input_data['predicted_trajectory']
        lat_decision = input_data['predicted_lat_decision']
        long_decision = input_data['predicted_long_decision']
        surface = np.zeros((600, 1200, 3),np.uint8)
        surface[:, :800] = rgb
        surface[:400,800:1200] = input_data['bev']
        # surface[400:600,800:1000] = trajectory
        surface[440:600,1000:1200] = trajectory[0:160,:]
        # surface[400:600,800:1000] = input_data['map_t1']
        # surface[400:600,1000:1200] = input_data['map_t2']
        surface[:150,:200] = input_data['rgb_left']
        surface[:150, 600:800] = input_data['rgb_right']
        surface[:150, 325:475] = input_data['rgb_focus']
        surface = cv2.putText(surface, input_data['language'], (20,580), cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0), 1)
        surface = cv2.putText(surface, input_data['control'], (20,560), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0), 1)
        surface = cv2.putText(surface, input_data['speed'], (20,540), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0), 1)
        surface = cv2.putText(surface, input_data['time'], (20,520), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0), 1)

        surface = cv2.putText(surface, 'Left  View', (40,135), cv2.FONT_HERSHEY_SIMPLEX,0.75,(0,0,0), 2)
        surface = cv2.putText(surface, 'Focus View', (335,135), cv2.FONT_HERSHEY_SIMPLEX,0.75,(0,0,0), 2)
        surface = cv2.putText(surface, 'Right View', (640,135), cv2.FONT_HERSHEY_SIMPLEX,0.75,(0,0,0), 2)

        surface = cv2.putText(surface, 'Behavior Decision', (820,420), cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0), 2)
        surface = cv2.putText(surface, 'Planned Trajectory', (1010,420), cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0), 2)
        surface = cv2.putText(surface, long_decision, (820,480), cv2.FONT_HERSHEY_SIMPLEX,0.75,(255,255,255), 2)
        surface = cv2.putText(surface, lat_decision, (820,530), cv2.FONT_HERSHEY_SIMPLEX,0.75,(255,255,255), 2)
        # surface = cv2.putText(surface, 'Future Prediction', (940,420), cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0), 2)
        # surface = cv2.putText(surface, 't', (1160,385), cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,0), 2)
        # surface = cv2.putText(surface, '0', (1170,385), cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0), 2)
        # surface = cv2.putText(surface, 't', (960,585), cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,0), 2)
        # surface = cv2.putText(surface, '1', (970,585), cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0), 2)
        # surface = cv2.putText(surface, 't', (1160,585), cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,0), 2)
        # surface = cv2.putText(surface, '2', (1170,585), cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0), 2)

        surface[:150,198:202]=0
        surface[:150,323:327]=0
        surface[:150,473:477]=0
        surface[:150,598:602]=0
        surface[148:152, :200] = 0
        surface[148:152, 325:475] = 0
        surface[148:152, 600:800] = 0
        surface[430:600, 998:1000] = 255
        surface[0:600, 798:800] = 255
        surface[0:600, 1198:1200] = 255
        surface[0:2, 800:1200] = 255
        surface[598:600, 800:1200] = 255
        surface[398:400, 800:1200] = 255


        # display image
        self._surface = pygame.surfarray.make_surface(surface.swapaxes(0, 1))
        if self._surface is not None:
            self._display.blit(self._surface, (0, 0))

        pygame.display.flip()
        pygame.event.get()
        return surface

    def _quit(self):
        pygame.quit()

def _location(x, y, z):
    return carla.Location(x=float(x), y=float(y), z=float(z))

def _orientation(yaw):
    return np.float32([np.cos(np.radians(yaw)), np.sin(np.radians(yaw))])

def get_collision(p1, v1, p2, v2):
    A = np.stack([v1, -v2], 1)
    b = p2 - p1

    if abs(np.linalg.det(A)) < 1e-3:
        return False, None

    x = np.linalg.solve(A, b)
    collides = all(x >= 0) and all(x <= 1)  # how many seconds until collision

    return collides, p1 + x[0] * v1

def _numpy(carla_vector, normalize=False):
    result = np.float32([carla_vector.x, carla_vector.y])

    if normalize:
        return result / (np.linalg.norm(result) + 1e-4)

    return result

def get_nearby_lights(vehicle, lights, pixels_per_meter=5.5, size=512, radius=5):
    result = list()

    transform = vehicle.get_transform()
    pos = transform.location
    theta = np.radians(90 + transform.rotation.yaw)
    R = np.array(
        [
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ]
    )

    for light in lights:
        delta = light.get_transform().location - pos

        target = R.T.dot([delta.x, delta.y])
        target *= pixels_per_meter
        target += size // 2

        if min(target) < 0 or max(target) >= size:
            continue

        trigger = light.trigger_volume
        light.get_transform().transform(trigger.location)
        dist = trigger.location.distance(vehicle.get_location())
        a = np.sqrt(
            trigger.extent.x**2 + trigger.extent.y**2 + trigger.extent.z**2
        )
        b = np.sqrt(
            vehicle.bounding_box.extent.x**2
            + vehicle.bounding_box.extent.y**2
            + vehicle.bounding_box.extent.z**2
        )

        if dist > a + b:
            continue

        result.append(light)

    return result

def get_entry_point():
    return "InterfuserAgent"

class Resize2FixedSize:
    def __init__(self, size):
        self.size = size

    def __call__(self, pil_img):
        pil_img = pil_img.resize(self.size)
        return pil_img

def create_carla_rgb_transform(
    input_size, need_scale=True, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD
):

    if isinstance(input_size, (tuple, list)):
        img_size = input_size[-2:]
    else:
        img_size = input_size
    tfl = []

    if isinstance(input_size, (tuple, list)):
        input_size_num = input_size[-1]
    else:
        input_size_num = input_size

    if need_scale:
        if input_size_num == 112:
            tfl.append(Resize2FixedSize((170, 128)))
        elif input_size_num == 128:
            tfl.append(Resize2FixedSize((195, 146)))
        elif input_size_num == 224:
            tfl.append(Resize2FixedSize((341, 256)))
        elif input_size_num == 256:
            tfl.append(Resize2FixedSize((288, 288)))
        else:
            raise ValueError("Can't find proper crop size")
    tfl.append(transforms.CenterCrop(img_size))
    tfl.append(transforms.ToTensor())
    tfl.append(transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)))

    return transforms.Compose(tfl)


class InterfuserAgent(autonomous_agent.AutonomousAgent):
# class InterfuserAgent(MapAgent):
    # for stop signs
    PROXIMITY_THRESHOLD = 30.0  # meters
    SPEED_THRESHOLD = 0.1
    WAYPOINT_STEP = 1.0  # meters
    
    def setup(self, path_to_conf_file):

        # self._hic = DisplayInterface()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print('use:', self.device)
        self.lidar_processed = list()
        self.track = autonomous_agent.Track.SENSORS
        self.step = -1
        self.control_step = 0 #-1
        self.wall_start = time.time()
        self.initialized = False
        self.rgb_front_transform = create_carla_rgb_transform(224)
        self.rgb_left_transform = create_carla_rgb_transform(128)
        self.rgb_right_transform = create_carla_rgb_transform(128)
        self.rgb_center_transform = create_carla_rgb_transform(128, need_scale=False)

        self.tracker = Tracker()

        self.config = imp.load_source("MainModel", path_to_conf_file).GlobalConfig(setting='eval')
        self.skip_frames = self.config.skip_frames
        self.seq_len = self.config.seq_len
        self.pred_len = self.config.pred_len
    
        self.img_states = np.zeros(shape=(self.seq_len, 3,
                                          self.config.img_resolution[0],
                                          self.config.img_resolution[1]),
                                    dtype=np.int32,
                                    )

        # self.img_states = np.zeros(shape=(self.seq_len, 3,
        #                                   224,
        #                                   224),
        #                            dtype=np.int32,
        #                            )
        
        # for LingoQA dataset
        # case_name = '0d37aee4-6508-33a2-998d-724834e80030' # bus station braking
        # case_name = '074d2237-ed1b-34d7-a2fc-68edbce50bb2' # pedestrain
        case_name = 'parking' # pedestrain
        # argo_path = '/media/spyder/94714162-4d32-4b72-b5d5-74dc17149d9c/autonomous_driving/argoverse2/train'
        argo_path = '/home/oscar/Dropbox/argoverse_dataset'

        image_list = []
        image_dir = f'{argo_path}/{case_name}/rgb_front'
        file_list = os.listdir(image_dir)
        file_list.sort()
        for image_name in file_list:
            image_list.append(cv2.cvtColor(cv2.imread(os.path.join(image_dir, image_name)), cv2.COLOR_BGR2RGB))

        # rgb = image_list[::-1]
        self.rgb_front =[]
        for idx, data in enumerate(image_list):
            self.rgb_front.append(data)

        self.target_waypoint = [np.ones(shape=(2)) * np.array([0.05, -20.0])] * 333
        ## for intersection left turn ##
        # self.target_waypoint = np.array(self.target_waypoint)
        x_value = 15 - 15/40*np.arange(1, 41)
        self.target_waypoint[50:91] = [np.array([x_value[i], -10.0]) for i in range(40)] # -10
        
        x_value = 15 - 15/55*np.arange(1, 56)
        self.target_waypoint[110:165] = [np.array([x_value[i], -10.0]) for i in range(55)] # -10

        x_value = 15 - 15/40*np.arange(1, 41)
        self.target_waypoint[180:220] = [np.array([x_value[i], -10.0]) for i in range(40)] # -10

        x_value = -10 + 10/40*np.arange(1, 41)
        self.target_waypoint[260:300] = [np.array([x_value[i], -10.0]) for i in range(40)] # -10

        ######################
        self.actions = np.zeros(shape=(0, self.config.act_dim))
        self.rtgs = np.ones(shape=(self.seq_len, 1)) * 30.0 #15.06
        self.decision_states =  None #np.zeros(shape=(0, len(RoadOption) - 1))
        self.detection_states = np.zeros(shape=(0, 5))
        self.timesteps = np.array(([0.0], [1.0], [2.0]))
        self.target_waypoints = np.zeros(shape=(0, 2))
        self.route_command = 4


        if self.config.model == 'GPT':
            self.net = init_gpt_vla(self.config)
        else:
            self.net = init_llama_vla(self.config)
        
        print("=============load=================")
        load_file = self.config.model_path
        self.net.load_state_dict(torch.load(load_file, map_location=torch.device('cpu'))["model"])
            
        self.net.cuda(device=self.device)
        self.net.eval()

        if self.net.prompt is not None:
            self.net.prompt.task_count = self.config.task_id - 1
        
        if self.config.model == 'GPT':
            self.net.prompt_gpt.vlm_adapter.random_instruction()
        
        if isinstance(self.config.model_path, list):
            self.ensemble = True
        else:
            self.ensemble = False

        self.softmax = torch.nn.Softmax(dim=1)
        self.traffic_meta_moving_avg = np.zeros((400, 7))
        self.momentum = self.config.momentum
        self.prev_lidar = None
        self.prev_speed_decision_list = deque(maxlen=5)
        self.prev_route_decision_list = deque(maxlen=5)
        self.prev_speed_decision = 0
        self.prev_route_decision = 3
        self.prev_surround_map = None
        self.reward_scale = 100

        control = carla.VehicleControl()
        control.steer = float(0.0)
        control.throttle = float(0.0)
        control.brake = float(0.0)
        self.prev_control = control

        self.save_path = None
        if SAVE_PATH is not None:
            now = datetime.datetime.now()
            string = pathlib.Path(os.environ["ROUTES"]).stem + "_"
            string += "_".join(
                map(
                    lambda x: "%02d" % x,
                    (now.month, now.day, now.hour, now.minute, now.second),
                )
            )

            # print(string)

            self.save_path = pathlib.Path(SAVE_PATH) / string
            self.save_path.mkdir(parents=True, exist_ok=False)
            (self.save_path / "meta").mkdir(parents=True, exist_ok=False)

    def _init(self):
        # self._route_planner = RoutePlanner(4.0, 50.0)
        # self._route_planner.set_route(self._global_plan, True)
        
        self._route_planner = RoutePlanner(7.5, 25.0, 257)
        self._route_planner.set_route(self._global_plan, True)
        
        self._waypoint_planner = RoutePlanner(4.0, 50)
        self._waypoint_planner.set_route(self._plan_gps_HACK, True)
        print(len(self._waypoint_planner.route))
        
        self.controller = CORLController(self.config)
        # self._turn_controller = PIDController(K_P=1.25, K_I=0.75, K_D=0.3, n=40)
        self._turn_controller = PIDController(K_P=1.4, K_I=0.75, K_D=0.3, n=40)
        # self._turn_controller = PIDController(K_P=1.8, K_I=0.75, K_D=1.2, n=40)
        self._speed_controller = PIDController(K_P=5.0, K_I=0.5, K_D=1.0, n=40)
        
        self.initialized = True
        self._hic = DisplayInterface()
        self._vehicle = CarlaDataProvider.get_hero_actor()
        self._world = self._vehicle.get_world()
        
        if self.config.weather != "none":
            weather = WEATHERS[self.config.weather]
            self._world.set_weather(weather)
        
        # settings = self._world.get_settings()
        # settings.fixed_delta_seconds = 0.25 #0.05
        # self._world.apply_settings(settings)
        
        self._map = self._world.get_map()
        self._actors = self._world.get_actors()

        self._traffic_lights = get_nearby_lights(
            self._vehicle, self._actors.filter("*traffic_light*")
        )
        
        lights_list = self._world.get_actors().filter("*traffic_light*")
        self._list_traffic_lights = []
        for light in lights_list:
            center, waypoints = self.get_traffic_light_waypoints(light)
            self._list_traffic_lights.append((light, center, waypoints))
        (
            self._list_traffic_waypoints,
            self._dict_traffic_lights,
        ) = self._gen_traffic_light_dict(self._list_traffic_lights)
    
        self.birdview_producer = BirdViewProducer(
            CarlaDataProvider.get_client(),  # carla.Client
            target_size=PixelDimensions(width=400, height=400),
            pixels_per_meter=4,
            crop_type=BirdViewCropType.FRONT_AND_REAR_AREA,
        )
        
        # for stop signs
        self._target_stop_sign = None  # the stop sign affecting the ego vehicle
        self._stop_completed = False  # if the ego vehicle has completed the stop sign
        self._affected_by_stop = (
            False  # if the ego vehicle is influenced by a stop sign
        )

    def set_global_plan(self, global_plan_gps, global_plan_world_coord):
        super().set_global_plan(global_plan_gps, global_plan_world_coord)

        self._plan_HACK = global_plan_world_coord
        self._plan_gps_HACK = global_plan_gps

    def get_traffic_light_waypoints(self, traffic_light):
        base_transform = traffic_light.get_transform()
        base_rot = base_transform.rotation.yaw
        area_loc = base_transform.transform(traffic_light.trigger_volume.location)

        # Discretize the trigger box into points
        area_ext = traffic_light.trigger_volume.extent
        x_values = np.arange(
            -0.9 * area_ext.x, 0.9 * area_ext.x, 1.0
        )  # 0.9 to avoid crossing to adjacent lanes

        area = []
        for x in x_values:
            point = self.rotate_point(carla.Vector3D(x, 0, area_ext.z), base_rot)
            point_location = area_loc + carla.Location(x=point.x, y=point.y)
            area.append(point_location)

        # Get the waypoints of these points, removing duplicates
        ini_wps = []
        for pt in area:
            wpx = self._map.get_waypoint(pt)
            # As x_values are arranged in order, only the last one has to be checked
            if (
                not ini_wps
                or ini_wps[-1].road_id != wpx.road_id
                or ini_wps[-1].lane_id != wpx.lane_id
            ):
                ini_wps.append(wpx)

        # Advance them until the intersection
        wps = []
        for wpx in ini_wps:
            while not wpx.is_intersection:
                next_wp = wpx.next(0.5)[0]
                if next_wp and not next_wp.is_intersection:
                    wpx = next_wp
                else:
                    break
            wps.append(wpx)

        return area_loc, wps

    def rotate_point(self, point, angle):
        """
        rotate a given point by a given angle
        """
        x_ = (
            math.cos(math.radians(angle)) * point.x
            - math.sin(math.radians(angle)) * point.y
        )
        y_ = (
            math.sin(math.radians(angle)) * point.x
            + math.cos(math.radians(angle)) * point.y
        )
        return carla.Vector3D(x_, y_, point.z)

    def _gen_traffic_light_dict(self, traffic_lights_list):
        traffic_light_dict = {}
        waypoints_list = []
        for light, center, waypoints in traffic_lights_list:
            for waypoint in waypoints:
                traffic_light_dict[waypoint] = (light, center)
                waypoints_list.append(waypoint)
        return waypoints_list, traffic_light_dict

    def _get_position(self, tick_data):
        gps = tick_data["gps"]
        gps = (gps - self._route_planner.mean) * self._route_planner.scale
        return gps

    def sensors(self):
        return [
            {
                "type": "sensor.camera.rgb",
                "x": 1.3,
                "y": 0.0,
                "z": 2.3,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
                "width": 800,
                "height": 600,
                "fov": 100,
                "id": "rgb",
            },
            {
                "type": "sensor.camera.rgb",
                "x": 1.3,
                "y": 0.0,
                "z": 2.3,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": -60.0,
                "width": 400,
                "height": 300,
                "fov": 100,
                "id": "rgb_left",
            },
            {
                "type": "sensor.camera.rgb",
                "x": 1.3,
                "y": 0.0,
                "z": 2.3,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 60.0,
                "width": 400,
                "height": 300,
                "fov": 100,
                "id": "rgb_right",
            },
            {
                "type": "sensor.lidar.ray_cast",
                "x": 1.3,
                "y": 0.0,
                "z": 2.5,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": -90.0,
                "id": "lidar",
            },
            {
                "type": "sensor.other.imu",
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
                "sensor_tick": 0.05,
                "id": "imu",
            },
            {
                "type": "sensor.other.gnss",
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
                "sensor_tick": 0.01,
                "id": "gps",
            },
            {"type": "sensor.speedometer", "reading_frequency": 20, "id": "speed"},
        ]

    def tick(self, input_data):

        rgb = cv2.cvtColor(input_data["rgb"][1][:, :, :3], cv2.COLOR_BGR2RGB)
        rgb_left = cv2.cvtColor(input_data["rgb_left"][1][:, :, :3], cv2.COLOR_BGR2RGB)
        rgb_right = cv2.cvtColor(
            input_data["rgb_right"][1][:, :, :3], cv2.COLOR_BGR2RGB
        )
        gps = input_data["gps"][1][:2]
        speed = input_data["speed"][1]["speed"]
        compass = input_data["imu"][1][-1]
        if (
            math.isnan(compass) == True
        ):  # It can happen that the compass sends nan for a few frames
            compass = 0.0

        result = {
            "rgb": rgb,
            "rgb_left": rgb_left,
            "rgb_right": rgb_right,
            "gps": gps,
            "speed": speed,
            "compass": compass,
        }

        pos = self._get_position(result)

        lidar_data = input_data['lidar'][1]
        result['raw_lidar'] = lidar_data

        lidar_unprocessed = lidar_data[:, :3]
        lidar_unprocessed[:, 1] *= -1
        full_lidar = transform_2d_points(
            lidar_unprocessed,
            np.pi / 2 - compass,
            -pos[0],
            -pos[1],
            np.pi / 2 - compass,
            -pos[0],
            -pos[1],
        )
        lidar_processed = lidar_to_histogram_features(full_lidar, crop=224)
        if self.step % 2 == 0 or self.step < 4:
            self.prev_lidar = lidar_processed
        result["lidar"] = self.prev_lidar

        result["gps"] = gps #pos
        result["pos"] = pos
        #######################################################################
        next_wp, next_cmd = self._waypoint_planner.run_step(pos)
        target_wp, target_cmd = self._route_planner.run_step(pos)
        #######################################################################
        
        result['next_wp'] = next_wp
        result["next_cmd"] = next_cmd.value
        result['target_wp'] = target_wp
        result['target_cmd'] = target_cmd.value
        result['measurements'] = [pos[0], pos[1], compass, speed]

        theta = compass + np.pi / 2
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

        local_command_point = np.array([target_wp[0] - pos[0], target_wp[1] - pos[1]])
        local_command_point = R.T.dot(local_command_point)
        result["target_local_waypoint"] = local_command_point

        # detections
        actors = self._world.get_actors()
        
        light = self._is_light_red(actors.filter("*traffic_light*"))
        walker = self._is_walker_hazard(actors.filter("*walker*"))
        bike = self._is_bike_hazard(actors.filter("*vehicle*"))
        stop_sign = self._is_stop_sign_hazard(actors.filter("*stop*"))
        vehicle = self._is_vehicle_hazard(actors.filter("*vehicle*"), self.route_command)
        junction_vehicle = self._is_junction_vehicle_hazard(actors.filter("*vehicle*"), self.route_command)
        # vehicle = self._is_vehicle_hazard(actors.filter("*vehicle*"), next_cmd)
        # junction_vehicle = self._is_junction_vehicle_hazard(actors.filter("*vehicle*"), next_cmd)
        vehicle = vehicle + junction_vehicle

        # record the reason for braking
        bike = [x.id for x in bike]
        pedestrian = [x.id for x in walker]
        red_light = [x.id for x in light]
        stop_sign = [x.id for x in stop_sign]
        vehicle = [x.id for x in vehicle]
        
        full_list = [vehicle, bike, pedestrian, red_light, stop_sign]
        detection = np.zeros((len(full_list)))
        for idx, item in enumerate(full_list):
            if len(item) > 0:
                # target_speed = 0.0
                detection[idx] = 0.0
            # else:
            #     target_speed = self.config.max_speed

        result['detection'] = detection

        return result

    @torch.no_grad()
    def run_step(self, input_data, timestamp):
        if not self.initialized:
            self._init()

        self.step += 1
        
        # if self.step % self.skip_frames != 0 and self.step > 4:
        #     return self.prev_control

        tick_data = self.tick(input_data)
        # pos = self._get_position(tick_data)

        velocity = tick_data["speed"]
        
        #######################################################################
        command = tick_data["next_cmd"]
        # command = self._command_planner.route[0][1].value
        #######################################################################
        
        timestep = np.array(self.step+3)[np.newaxis]
        detection = tick_data['detection']
        decision = None
        # target_waypoint, target_command = tick_data['target_local_waypoint'], None

        # rgb = cv2.resize(tick_data["rgb"], tuple(self.config.img_resolution))

        start_time = time.time()

        # for lingoqa dataset
        target_command = None
        target_waypoint = self.target_waypoint.pop(0)
        self.rgb_front_current = self.rgb_front.pop(0)
        self.rgb_left_current = self.rgb_front_current
        self.rgb_right_current = self.rgb_front_current
        self.rgb_rear_left_current = self.rgb_front_current

        rgb = cv2.resize(self.rgb_front_current, tuple(self.config.img_resolution))

        if self.config.image_type == 'segmentation':
            segmentation_image = self.net.seg_encoder.forward(rgb)

        image = np.transpose(segmentation_image, (2, 0, 1))

        self.states_adapter(image, detection, timestep, decision_states=decision, target_waypoints=target_waypoint)
                
        if self.config.model == 'GPT': #GPTVLA
            img_states = torch.tensor(self.img_states, dtype=torch.bfloat16).to(device=self.device)
            img_states = img_states / 255. #bfloat16 (4,3,224,224)
            detection_states = torch.tensor(self.detection_states, dtype=torch.bfloat16).unsqueeze(0).to(device=self.device) #bfloat16 (1,4,5)
            actions = torch.tensor(self.actions, dtype=torch.bfloat16)
            actions = torch.cat([actions, torch.zeros((1, self.config.act_dim), dtype=torch.bfloat16)], dim=0).unsqueeze(0).to(device=self.device) # bfloat16 (1,4,9)
            target_waypoints = torch.tensor(self.target_waypoints, dtype=torch.bfloat16).unsqueeze(0).to(device=self.device) #bfloat16 (1,4,2)
            rtgs = torch.tensor(self.rtgs, dtype=torch.bfloat16).unsqueeze(0).to(device=self.device) #bfloat16 (1,4,1)
            timesteps = torch.tensor(self.timesteps, dtype=torch.int32).unsqueeze(0).to(device=self.device) #torch.int32 (1,4,1)
        else: # MobileVLA
            img_states = torch.tensor(self.img_states, dtype=torch.float16).to(device=self.device)
            detection_states = torch.tensor(self.detection_states, dtype=torch.float16).unsqueeze(0).to(device=self.device)
            actions = torch.tensor(self.actions, dtype=torch.float16)
            actions = torch.cat([actions, torch.zeros((1, self.config.act_dim), dtype=torch.float32)], dim=0).unsqueeze(0).to(device=self.device)
            target_waypoints = torch.tensor(self.target_waypoints, dtype=torch.float16).unsqueeze(0).to(device=self.device)
            rtgs = torch.tensor(self.rtgs, dtype=torch.float16).unsqueeze(0).to(device=self.device)
            timesteps = torch.tensor(self.timesteps, dtype=torch.int32).unsqueeze(0).to(device=self.device)

        ### bus station braking ####
        # if len(self.rgb_front) < 90:
        #     detection_states[:,:,0] = 1.0

        ### pedetrain ###
        # if len(self.rgb_front) < 160 and len(self.rgb_front) > 20:
        #     detection_states[:,:,2] = 1.0

        ### NTU B4 ####
        if len(self.rgb_front) > 122 and len(self.rgb_front) < 129:
            detection_states[:,:,0] = 1.0

        if decision is None:
            decision_states = None
        else:
            decision_states = torch.tensor(self.decision_states, dtype=torch.float64).unsqueeze(0).to(device=self.device)
        
        # if self.ensemble:
        #     outputs = []
        #     with torch.no_grad():
        #         for net in self.nets:
        #             action, _ = net.get_action(img_states, actions, rtgs, timesteps, detection_states,
        #                          decision_states, target_waypoints=target_waypoints)
        #             outputs.append(action)
        # else:
        with torch.no_grad():
           actions, actions_one_hot, trajectory, route_logit, speed_logit = self.net.get_action(img_states, actions, rtgs, timesteps, detection_states,
                                                                                                decisions=decision_states, target_waypoints=target_waypoints)

        # action = torch.nn.functional.one_hot(actions.squeeze(0)[-1], num_classes=6).detach().cpu().numpy()
        # next_command = actions.squeeze(0)[-1].detach().cpu().numpy() + 1
        
        action_route = actions[0]
        action_speed = actions[-1]

        next_route_command = action_route.squeeze(0)[-1].detach().cpu().numpy() + 1 #adjust the number to the enum of RoadOption
        next_speed_command = action_speed.squeeze(0)[-1].detach().cpu().numpy()
        
        # if np.all(np.array(self.prev_speed_decision_list) == next_speed_command):
        #     self.prev_speed_decision = next_speed_command
        #     self.prev_speed_decision_list.append(next_speed_command)  
        # else:
        #     self.prev_speed_decision_list.append(next_speed_command)  
        #     next_speed_command = self.prev_speed_decision

        # if np.all(np.array(self.prev_route_decision_list) == next_route_command):
        #     self.prev_route_decision = next_route_command
        #     self.prev_route_decision_list.append(next_route_command)  
        # else:
        #     self.prev_route_decision_list.append(next_route_command)  
        #     next_route_command = self.prev_route_decision 

        next_command = [next_route_command, next_speed_command]
        self.route_command = next_route_command

        action = actions_one_hot.squeeze(0).detach().cpu().numpy()
        trajectory = trajectory.squeeze(0).detach().cpu().numpy()
        next_node = trajectory[2,:]
                
        target_node = target_waypoint
        target_command = target_command

        ##### Safe Controller #####
        steer, throttle, brake, target_speed = self._get_control(
            next_node, target_node, next_command, tick_data
        )
        
        self.is_junction = self._map.get_waypoint(
            self._vehicle.get_location()
        ).is_junction

        light = self._find_closest_valid_traffic_light(
            self._vehicle.get_location(), min_dis=50
        )
        if light is not None:
            self.affected_light_id = light.id
        else:
            self.affected_light_id = -1
        
        # print("decision:", decision, "detection:", detection)
        
        control = carla.VehicleControl()
        control.steer = steer #+ 1e-2 * np.random.randn()
        control.throttle = throttle
        control.brake = float(brake)
        
        ##### reward #####
        # action reward
        if abs(steer - self.prev_control.steer) > 0.01:
            r_action = -0.1
        else:
            r_action = 0.0
        
        # speed reward

        r_speed = 1.0 - np.abs(velocity - target_speed) / self.config.max_speed

        # lateral displacement reward
        ego_theta = tick_data["compass"]
        pos = tick_data["pos"]
        ego_x = pos[0]
        ego_y = pos[1]
        loc = self._vehicle.get_location()
        ego_waypoint = self._map.get_waypoint(loc)
        wp_x = - ego_waypoint.transform.location.y
        wp_y = ego_waypoint.transform.location.x
        R = np.array([
            [np.cos(np.pi/2+ego_theta), -np.sin(np.pi/2+ego_theta)],
            [np.sin(np.pi/2+ego_theta),  np.cos(np.pi/2+ego_theta)]
            ])
        ego_waypoint_local = R.T.dot(np.array([wp_x - ego_x, wp_y - ego_y]))
        
        r_position = ego_waypoint_local[-1]
        reward = r_speed + r_position + r_action
        
        self.actions_rewards_adapter(action, reward)

        total_time = time.time() - start_time
        print('One step time:', total_time)

        self_car_map = render_self_car(
            loc=np.array([0, 0]),
            ori=np.array([0, -1]),
            box=np.array([2.45, 1.0]),
            color=[1, 1, 0], pixels_per_meter=10, max_distance=10,
        )

        render_trajectory = render_waypoints(trajectory, pixels_per_meter=30, max_distance=15, color=(0, 255, 0))

        self_car_map = cv2.resize(self_car_map, (200, 200))
        render_trajectory = cv2.resize(render_trajectory, (200, 200))

        surround_map = np.clip(
            (
                self_car_map.astype(np.float32)
                + render_trajectory.astype(np.float32)
            ),
            0,
            255,
        ).astype(np.uint8)
        tick_data["predicted_trajectory"] = surround_map
        tick_data["predicted_lat_decision"] = LAT_DECISION_LIST[next_route_command-1]
        
        if self.should_brake:
            tick_data["predicted_long_decision"] = 'Braking'
        elif self.should_slow:
            tick_data["predicted_long_decision"] = 'Slow Down'
        else:
            tick_data["predicted_long_decision"] = 'Pedal'

        tick_data["rgb_raw"] = tick_data["rgb"]
        tick_data["rgb_left_raw"] = tick_data["rgb_left"]
        tick_data["rgb_right_raw"] = tick_data["rgb_right"]

        tick_data["rgb"] = cv2.resize(self.rgb_front_current, (800, 600))
        tick_data["rgb_left"] = cv2.resize(self.rgb_front_current, (200, 150))
        tick_data["rgb_right"] = cv2.resize(self.rgb_front_current, (200, 150))
        tick_data["rgb_focus"] = cv2.resize(self.rgb_front_current, (150, 150))
        tick_data["control"] = "throttle: %.2f, steer: %.2f, brake: %.2f" % (
            control.throttle,
            control.steer,
            control.brake,
        )
        tick_data["speed"] = "speed: %.2f Km/h, target_speed: %.2f Km/h" % (velocity*3.6, target_speed*3.6)
        tick_data["language"] = "Instruction:" + self.net.prompt_vlm.vlm_adapter.prompt_list[-1]
        bev = tick_data["raw_lidar"]
        bev = BirdViewProducer.as_rgb(
            self.birdview_producer.produce(agent_vehicle=self._vehicle)
        )
                
        tick_data["bev"] = cv2.resize(segmentation_image, (400, 400))

        tick_data["mes"] = "speed: %.2f" % velocity
        tick_data["time"] = "time: %.3f" % timestamp
        surface = self._hic.run_interface(tick_data)
        tick_data["surface"] = surface
            
        next_wp = tick_data['next_wp']
        next_cmd = tick_data["next_cmd"]
        predicted_wp, _, _, _=np.linalg.lstsq(R.T, next_node)
        predicted_wp = predicted_wp + np.array((ego_x, ego_y))
        self.prev_control = control
        
        return control

    def save(self, tick_data):
        frame = self.step // self.skip_frames
        Image.fromarray(tick_data["surface"]).save(
            self.save_path / "meta" / ("%04d.jpg" % frame)
        )
        return

    def destroy(self):
        if self.ensemble:
            del self.nets
        else:
            del self.net

    def actions_rewards_adapter(self, actions, rewards):
        self.actions = np.concatenate((self.actions, actions[np.newaxis]), axis=0)[-self.seq_len:]
        self.rtgs = np.concatenate((self.rtgs, (self.rtgs[-1] - rewards/self.reward_scale)[np.newaxis]), axis=0)[-self.seq_len:]

    def states_adapter(self, img_states, detection_states, timesteps, decision_states=None, target_waypoints=None):
        # print("before timesteps:", self.timesteps)
        self.img_states = np.concatenate((self.img_states, img_states[np.newaxis]), axis=0)[-self.seq_len:]
        self.detection_states = np.concatenate((self.detection_states, detection_states[np.newaxis]), axis=0)[-self.seq_len:]
        self.timesteps = np.concatenate((self.timesteps, timesteps[np.newaxis]), axis=0)[-self.seq_len:]

        if decision_states is not None:
            self.decision_states = np.concatenate((self.decision_states, decision_states[np.newaxis]), axis=0)[-self.seq_len:]

        if target_waypoints is not None:
            self.target_waypoints = np.concatenate((self.target_waypoints, target_waypoints[np.newaxis]), axis=0)[-self.seq_len:]

    def _get_angle_to(self, pos, theta, target):
        R = np.array(
            [
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)],
            ]
        )

        aim = R.T.dot(target - pos)
        angle = -np.degrees(np.arctan2(-aim[1], aim[0]))
        angle = 0.0 if np.isnan(angle) else angle

        return angle

    def _tracking_control(self, target, target_speed, tick_data):
        speed = tick_data["speed"]

        # Steering.
        angle = -np.degrees(np.arctan2(-target[0], -target[1]))
        # angle = np.degrees(np.pi/2 - np.arctan2(-target[0], -target[1]))
        angle_unnorm = 0.0 if np.isnan(angle) else angle
        angle = angle_unnorm / 90

        steer = self._turn_controller.step(angle)
        steer = np.clip(steer, -1.0, 1.0)
        steer = round(steer, 3)

        if target_speed < 0.1:
            brake = True
            throttle = 0.0
            self.should_brake = True
            self.should_slow = False
        else:
            delta = np.clip(target_speed - speed, 0.0, 0.25)
            throttle = self._speed_controller.step(delta)
            throttle = np.clip(throttle, 0.0, 0.75)
            brake = False
            self.should_brake = False
            
            if target_speed < 5.0:
                self.should_slow = True
            else:
                self.should_slow = False

        return steer, throttle, brake, target_speed

    def _get_control(self, target, far_target, near_command, tick_data):
        speed = tick_data["speed"]

        route_command = near_command[0]
        speed_command = near_command[1]

        # Steering.
        angle = -np.degrees(np.arctan2(-target[0], -target[1]))
        # angle = np.degrees(np.pi/2 - np.arctan2(-target[0], -target[1]))
        angle_unnorm = 0.0 if np.isnan(angle) else angle
        angle = angle_unnorm / 90

        steer = self._turn_controller.step(angle)
        steer = np.clip(steer, -1.0, 1.0)
        steer = round(steer, 3)

        # Acceleration.
        angle = -np.degrees(np.arctan2(-far_target[0], -far_target[1]))
        # angle = np.degrees(np.pi/2 - np.arctan2(-far_target[0], -far_target[1]))
        angle_far_unnorm = 0.0 if np.isnan(angle) else angle
        
        should_slow = abs(angle_far_unnorm) > 45.0 or abs(angle_unnorm) > 5.0
        self.should_slow = should_slow
        target_speed = 4.0 if should_slow else 6.5
        # target_speed = 8.0 if should_slow else 12
        brake = self._should_brake(route_command)
        target_speed = target_speed if not brake else 0.0
        self.should_brake = brake
        
        # print('Rule Speed Decision:', target_speed)

        speed_decision = LONG_DECISION_LIST[speed_command]
        
        if speed_decision == 'Brake':
            target_speed = 0.0
            self.should_brake = 1
            self.should_slow = 0
        elif speed_decision == 'Slow Down':
            target_speed = 4.0
            self.should_slow = 1
            self.should_brake = 0
        else:
            self.should_slow = 0
            self.should_brake = 0
            target_speed = 6.5
            
        # print('VLMDrive-Pro Speed Decision:', target_speed)

        delta = np.clip(target_speed - speed, 0.0, 0.25)
        throttle = self._speed_controller.step(delta)
        throttle = np.clip(throttle, 0.0, 0.75)

        if self.should_brake:
            # steer *= 0.5
            # print('brake:', brake)
            throttle = 0.0

        return steer, throttle, self.should_brake, target_speed

    def _control(self, target, far_target, near_command, far_command, tick_data):

        pos = self._get_position(tick_data)
        theta = tick_data["compass"]
        speed = tick_data["speed"]

        # Steering.
        angle_unnorm = self._get_angle_to(pos, theta, target)
        angle = angle_unnorm / 90

        steer = self._turn_controller.step(angle)
        steer = np.clip(steer, -1.0, 1.0)
        steer = round(steer, 3)

        # Acceleration.
        angle_far_unnorm = self._get_angle_to(pos, theta, far_target)
        should_slow = abs(angle_far_unnorm) > 45.0 or abs(angle_unnorm) > 5.0
        self.should_slow = should_slow
        target_speed = 4.0 if should_slow else 6.5
        brake = self._should_brake(near_command)
        self.should_brake = brake
        target_speed = target_speed if not brake else 0.0

        delta = np.clip(target_speed - speed, 0.0, 0.25)
        throttle = self._speed_controller.step(delta)
        throttle = np.clip(throttle, 0.0, 0.75)

        if brake:
            # steer *= 0.5
            throttle = 0.0

        return steer, throttle, brake, target_speed

    def _should_brake(self, command):
        actors = self._world.get_actors()

        vehicle = self._is_vehicle_hazard(actors.filter("*vehicle*"), command)
        lane_vehicle = self._is_lane_vehicle_hazard(actors.filter("*vehicle*"), command)
        junction_vehicle = self._is_junction_vehicle_hazard(
            actors.filter("*vehicle*"), command
        )
        red_light = self._is_light_red(actors.filter("*traffic_light*"))
        pedestrian = self._is_walker_hazard(actors.filter("*walker*"))
        bike = self._is_bike_hazard(actors.filter("*vehicle*"))
        stop_sign = self._is_stop_sign_hazard(actors.filter("*stop*"))

        # record the reason for braking
        self.is_vehicle_present = [x.id for x in vehicle]
        self.is_lane_vehicle_present = [x.id for x in lane_vehicle]
        self.is_junction_vehicle_present = [x.id for x in junction_vehicle]
        self.is_pedestrian_present = [x.id for x in pedestrian]
        self.is_bike_present = [x.id for x in bike]
        self.is_red_light_present = [x.id for x in red_light]
        self.is_stop_sign_present = [x.id for x in stop_sign]

        full_list = [bike, pedestrian, red_light, stop_sign]
        detection = np.zeros((len(full_list)))
        for idx, item in enumerate(full_list):
            if len(item) > 0:
                # target_speed = 0.0
                detection[idx] = 1.0
            # else:
                # target_speed = self.config.max_speed

        return any(len(x) > 0
            for x in [
                vehicle,
                lane_vehicle,
                junction_vehicle,
                bike,
                red_light,
                pedestrian,
                stop_sign,
            ]
        )

    def _is_vehicle_hazard(self, vehicle_list, command):
        res = []
        z = self._vehicle.get_location().z

        o1 = _orientation(self._vehicle.get_transform().rotation.yaw)
        p1 = _numpy(self._vehicle.get_location())
        s1 = max(
            10, 3.0 * np.linalg.norm(_numpy(self._vehicle.get_velocity()))
        )  # increases the threshold distance
        s1a = np.linalg.norm(_numpy(self._vehicle.get_velocity()))
        w1 = self._map.get_waypoint(self._vehicle.get_location())
        v1_hat = o1
        v1 = s1 * v1_hat

        for target_vehicle in vehicle_list:
            if target_vehicle.id == self._vehicle.id:
                continue
            if not target_vehicle.is_alive:
                continue

            o2 = _orientation(target_vehicle.get_transform().rotation.yaw)
            p2 = _numpy(target_vehicle.get_location())
            s2 = max(5.0, 2.0 * np.linalg.norm(_numpy(target_vehicle.get_velocity())))
            s2a = np.linalg.norm(_numpy(target_vehicle.get_velocity()))
            w2 = self._map.get_waypoint(target_vehicle.get_location())
            v2_hat = o2
            v2 = s2 * v2_hat

            p2_p1 = p2 - p1
            distance = np.linalg.norm(p2_p1)
            p2_p1_hat = p2_p1 / (distance + 1e-4)

            angle_to_car = np.degrees(np.arccos(np.clip(v1_hat.dot(p2_p1_hat), -1, 1)))
            angle_between_heading = np.degrees(np.arccos(np.clip(o1.dot(o2), -1, 1)))

            # to consider -ve angles too
            angle_to_car = min(angle_to_car, 360.0 - angle_to_car)
            angle_between_heading = min(
                angle_between_heading, 360.0 - angle_between_heading
            )

            if (
                not w2.is_junction
                and angle_between_heading > 45.0
                and s2a < 0.5
                and distance > 4
            ):
                if w1.road_id != w2.road_id:
                    continue
            if (angle_between_heading < 15
                and w1.road_id == w2.road_id
                and w1.lane_id != w2.lane_id
                and command != RoadOption.CHANGELANELEFT
                and command != RoadOption.CHANGELANERIGHT
            ):
                continue

            if angle_between_heading > 60.0 and not (
                angle_to_car < 15 and distance < s1
            ):
                continue
            elif angle_to_car > 30.0:
                continue
            elif distance > s1:
                continue

            res.append(target_vehicle)

        return res


    def _is_lane_vehicle_hazard(self, vehicle_list, command):
        res = []
        if (
            command != RoadOption.CHANGELANELEFT
            and command != RoadOption.CHANGELANERIGHT
        ):
            return []

        z = self._vehicle.get_location().z
        w1 = self._map.get_waypoint(self._vehicle.get_location())
        o1 = _orientation(self._vehicle.get_transform().rotation.yaw)
        p1 = self._vehicle.get_location()

        yaw_w1 = w1.transform.rotation.yaw
        lane_width = w1.lane_width
        location_w1 = w1.transform.location

        lft_shift = 0.5
        rgt_shift = 0.5
        if command == RoadOption.CHANGELANELEFT:
            rgt_shift += 1
        else:
            lft_shift += 1

        lft_lane_wp = self.rotate_point(
            carla.Vector3D(lft_shift * lane_width, 0.0, location_w1.z), yaw_w1 + 90
        )
        lft_lane_wp = location_w1 + carla.Location(lft_lane_wp)
        rgt_lane_wp = self.rotate_point(
            carla.Vector3D(rgt_shift * lane_width, 0.0, location_w1.z), yaw_w1 - 90
        )
        rgt_lane_wp = location_w1 + carla.Location(rgt_lane_wp)

        for target_vehicle in vehicle_list:
            if target_vehicle.id == self._vehicle.id:
                continue

            w2 = self._map.get_waypoint(target_vehicle.get_location())
            o2 = _orientation(target_vehicle.get_transform().rotation.yaw)
            p2 = target_vehicle.get_location()
            x2 = target_vehicle.bounding_box.extent.x
            p2_hat = p2 - target_vehicle.get_transform().get_forward_vector() * x2 * 2
            s2 = (
                target_vehicle.get_velocity()
                + target_vehicle.get_transform().get_forward_vector() * x2
            )
            s2_value = max(
                12,
                2
                + 2 * x2
                + 3.0 * np.linalg.norm(_numpy(target_vehicle.get_velocity())),
            )

            distance = p1.distance(p2)

            if distance > s2_value:
                continue
            if w1.road_id != w2.road_id or w1.lane_id * w2.lane_id < 0:
                continue
            if command == RoadOption.CHANGELANELEFT:
                if w1.lane_id > 0:
                    if w2.lane_id != w1.lane_id - 1:
                        continue
                if w1.lane_id < 0:
                    if w2.lane_id != w1.lane_id + 1:
                        continue
            if command == RoadOption.CHANGELANERIGHT:
                if w1.lane_id > 0:
                    if w2.lane_id != w1.lane_id + 1:
                        continue
                if w1.lane_id < 0:
                    if w2.lane_id != w1.lane_id - 1:
                        continue

            if self._are_vehicles_crossing_future(p2_hat, s2, lft_lane_wp, rgt_lane_wp):
                res.append(target_vehicle)
        return res
    
    def _is_junction_vehicle_hazard(self, vehicle_list, command):
        res = []
        o1 = _orientation(self._vehicle.get_transform().rotation.yaw)
        x1 = self._vehicle.bounding_box.extent.x
        p1 = (
            self._vehicle.get_location()
            + x1 * self._vehicle.get_transform().get_forward_vector()
        )
        w1 = self._map.get_waypoint(p1)
        s1 = np.linalg.norm(_numpy(self._vehicle.get_velocity()))
        if command == RoadOption.RIGHT:
            shift_angle = 25
        elif command == RoadOption.LEFT:
            shift_angle = -25
        else:
            shift_angle = 0
        v1 = (4 * s1 + 5) * _orientation(
            self._vehicle.get_transform().rotation.yaw + shift_angle
        )

        for target_vehicle in vehicle_list:
            if target_vehicle.id == self._vehicle.id:
                continue

            o2 = _orientation(target_vehicle.get_transform().rotation.yaw)
            o2_left = _orientation(target_vehicle.get_transform().rotation.yaw - 15)
            o2_right = _orientation(target_vehicle.get_transform().rotation.yaw + 15)
            x2 = target_vehicle.bounding_box.extent.x

            p2 = target_vehicle.get_location()
            p2_hat = p2 - (x2 + 2) * target_vehicle.get_transform().get_forward_vector()
            w2 = self._map.get_waypoint(p2)
            s2 = np.linalg.norm(_numpy(target_vehicle.get_velocity()))

            v2 = (4 * s2 + 2 * x2 + 6) * o2
            v2_left = (4 * s2 + 2 * x2 + 6) * o2_left
            v2_right = (4 * s2 + 2 * x2 + 6) * o2_right

            angle_between_heading = np.degrees(np.arccos(np.clip(o1.dot(o2), -1, 1)))

            # if self._vehicle.get_location().distance(p2) > 10: #non-conservative
            if self._vehicle.get_location().distance(p2) > 20: #conservative policy
                continue
            if w1.is_junction == False and w2.is_junction == False:
                continue
            if angle_between_heading < 15.0 or angle_between_heading > 165: # conservative policy
            # if angle_between_heading < 40.0 or angle_between_heading > 140: # non-conservative
            # if angle_between_heading < 40.0 or angle_between_heading > 120:
                continue
            collides, collision_point = get_collision(
                _numpy(p1), v1, _numpy(p2_hat), v2
            )
            if collides is None:
                collides, collision_point = get_collision(
                    _numpy(p1), v1, _numpy(p2_hat), v2_left
                )
            if collides is None:
                collides, collision_point = get_collision(
                    _numpy(p1), v1, _numpy(p2_hat), v2_right
                )

            light = self._find_closest_valid_traffic_light(
                target_vehicle.get_location(), min_dis=10
            )
            if (
                light is not None
                and light.state != carla.libcarla.TrafficLightState.Green
            ):
                continue
            if collides:
                # print('junction vehicle:', collides, collision_point)
                res.append(target_vehicle)
        return res
    
    def _is_light_red(self, lights_list):
        if (
            self._vehicle.get_traffic_light_state()
            != carla.libcarla.TrafficLightState.Green
        ):
            affecting = self._vehicle.get_traffic_light()

            for light in self._traffic_lights:
                if light.id == affecting.id:
                    return [light]

        light = self._find_closest_valid_traffic_light(
            self._vehicle.get_location(), min_dis=8
        )
        if light is not None and light.state != carla.libcarla.TrafficLightState.Green:
            return [light]
        return []

    def _find_closest_valid_traffic_light(self, loc, min_dis):
        wp = self._map.get_waypoint(loc)
        min_wp = None
        min_distance = min_dis
        for waypoint in self._list_traffic_waypoints:
            if waypoint.road_id != wp.road_id or waypoint.lane_id * wp.lane_id < 0:
                continue
            dis = loc.distance(waypoint.transform.location)
            if dis <= min_distance:
                min_distance = dis
                min_wp = waypoint
        if min_wp is None:
            return None
        else:
            return self._dict_traffic_lights[min_wp][0]

    def _is_walker_hazard(self, walkers_list):
        res = []
        p1 = _numpy(self._vehicle.get_location())
        v1 = 10.0 * _orientation(self._vehicle.get_transform().rotation.yaw)

        for walker in walkers_list:
            v2_hat = _orientation(walker.get_transform().rotation.yaw)
            s2 = np.linalg.norm(_numpy(walker.get_velocity()))

            if s2 < 0.05:
                v2_hat *= s2

            p2 = -3.0 * v2_hat + _numpy(walker.get_location())
            v2 = 8.0 * v2_hat

            collides, collision_point = get_collision(p1, v1, p2, v2)

            if collides:
                # print('walker:', collides, collision_point)
                res.append(walker)

        return res

    def _is_bike_hazard(self, bikes_list):
        res = []
        o1 = _orientation(self._vehicle.get_transform().rotation.yaw)
        v1_hat = o1
        p1 = _numpy(self._vehicle.get_location())
        v1 = 10.0 * o1

        for bike in bikes_list:
            o2 = _orientation(bike.get_transform().rotation.yaw)
            s2 = np.linalg.norm(_numpy(bike.get_velocity()))
            v2_hat = o2
            p2 = _numpy(bike.get_location())

            p2_p1 = p2 - p1
            distance = np.linalg.norm(p2_p1)
            p2_p1_hat = p2_p1 / (distance + 1e-4)

            angle_to_car = np.degrees(np.arccos(np.clip(v1_hat.dot(p2_p1_hat), -1, 1)))
            angle_between_heading = np.degrees(np.arccos(np.clip(o1.dot(o2), -1, 1)))

            # to consider -ve angles too
            angle_to_car = min(angle_to_car, 360.0 - angle_to_car)
            angle_between_heading = min(
                angle_between_heading, 360.0 - angle_between_heading
            )
            if distance > 10: #20:
                continue
            if angle_to_car > 15: #30:
                continue
            if angle_between_heading < 80 or angle_between_heading > 100:
                continue

            p2_hat = -2.0 * v2_hat + _numpy(bike.get_location())
            v2 = 7.0 * v2_hat

            collides, collision_point = get_collision(p1, v1, p2_hat, v2)

            if collides:
                # print('bike:', collides, collision_point)
                res.append(bike)

        return res

    def _is_stop_sign_hazard(self, stop_sign_list):
        res = []
        if self._affected_by_stop:
            if not self._stop_completed:
                current_speed = self._get_forward_speed()
                if current_speed < self.SPEED_THRESHOLD:
                    self._stop_completed = True
                    return res
                else:
                    return [self._target_stop_sign]
            else:
                # reset if the ego vehicle is outside the influence of the current stop sign
                if not self._is_actor_affected_by_stop(
                    self._vehicle, self._target_stop_sign
                ):
                    self._affected_by_stop = False
                    self._stop_completed = False
                    self._target_stop_sign = None
                return res

        ve_tra = self._vehicle.get_transform()
        ve_dir = ve_tra.get_forward_vector()

        wp = self._map.get_waypoint(ve_tra.location)
        wp_dir = wp.transform.get_forward_vector()

        dot_ve_wp = ve_dir.x * wp_dir.x + ve_dir.y * wp_dir.y + ve_dir.z * wp_dir.z

        if dot_ve_wp > 0:  # Ignore all when going in a wrong lane
            for stop_sign in stop_sign_list:
                if self._is_actor_affected_by_stop(self._vehicle, stop_sign):
                    # this stop sign is affecting the vehicle
                    self._affected_by_stop = True
                    self._target_stop_sign = stop_sign
                    res.append(self._target_stop_sign)

        return res


    def _is_actor_affected_by_stop(self, actor, stop, multi_step=20):
        """
        Check if the given actor is affected by the stop
        """
        affected = False
        # first we run a fast coarse test
        current_location = actor.get_location()
        stop_location = stop.get_transform().location
        if stop_location.distance(current_location) > self.PROXIMITY_THRESHOLD:
            return affected

        stop_t = stop.get_transform()
        transformed_tv = stop_t.transform(stop.trigger_volume.location)

        # slower and accurate test based on waypoint's horizon and geometric test
        list_locations = [current_location]
        waypoint = self._map.get_waypoint(current_location)
        for _ in range(multi_step):
            if waypoint:
                waypoint = waypoint.next(self.WAYPOINT_STEP)[0]
                if not waypoint:
                    break
                list_locations.append(waypoint.transform.location)

        for actor_location in list_locations:
            if self._point_inside_boundingbox(
                actor_location, transformed_tv, stop.trigger_volume.extent
            ):
                affected = True

        return affected

    def _point_inside_boundingbox(self, point, bb_center, bb_extent):
        A = carla.Vector2D(bb_center.x - bb_extent.x, bb_center.y - bb_extent.y)
        B = carla.Vector2D(bb_center.x + bb_extent.x, bb_center.y - bb_extent.y)
        D = carla.Vector2D(bb_center.x - bb_extent.x, bb_center.y + bb_extent.y)
        M = carla.Vector2D(point.x, point.y)

        AB = B - A
        AD = D - A
        AM = M - A
        am_ab = AM.x * AB.x + AM.y * AB.y
        ab_ab = AB.x * AB.x + AB.y * AB.y
        am_ad = AM.x * AD.x + AM.y * AD.y
        ad_ad = AD.x * AD.x + AD.y * AD.y

        return am_ab > 0 and am_ab < ab_ab and am_ad > 0 and am_ad < ad_ad

    def _get_forward_speed(self, transform=None, velocity=None):
        """Convert the vehicle transform directly to forward speed"""
        if not velocity:
            velocity = self._vehicle.get_velocity()
        if not transform:
            transform = self._vehicle.get_transform()

        vel_np = np.array([velocity.x, velocity.y, velocity.z])
        pitch = np.deg2rad(transform.rotation.pitch)
        yaw = np.deg2rad(transform.rotation.yaw)
        orientation = np.array(
            [np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), np.sin(pitch)]
        )
        speed = np.dot(vel_np, orientation)
        return speed