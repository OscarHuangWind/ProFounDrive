#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 14:48:46 2023

@author: oscar
"""
import sys
sys.path.append('home/oscar/Dropbox/InterFuser')

import os
import json
import time
import datetime
import pathlib
import time
import imp
import cv2
from collections import deque

import carla
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from carla_birdeye_view import BirdViewProducer, BirdViewCropType, PixelDimensions

import torch
import numpy as np
from PIL import Image
from easydict import EasyDict

from torchvision import transforms
from leaderboard.autoagents import autonomous_agent
from timm.models import create_model
from team_code.map_agent import MapAgent
from team_code.utils import lidar_to_histogram_features, transform_2d_points
from team_code.planner import RoutePlanner
from team_code.interfuser_controller import InterfuserController
from team_code.render import render, render_self_car, render_waypoints
from team_code.tracker import Tracker

from agents.navigation.local_planner import RoadOption

# from corl_config import GlobalConfig
from foundation_model.zoo import vit_pt_imnet

import math
import yaml

try:
    import pygame
except ImportError:
    raise RuntimeError("cannot import pygame, make sure pygame package is installed")


SAVE_PATH = os.environ.get("SAVE_PATH", 'eval')
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


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

        pygame.display.set_caption("Human Agent")

    def run_interface(self, input_data):
        rgb = input_data['rgb']
        rgb_left = input_data['rgb_left']
        rgb_right = input_data['rgb_right']
        rgb_focus = input_data['rgb_focus']
        # map = input_data['map']
        surface = np.zeros((600, 1200, 3),np.uint8)
        surface[:, :800] = rgb
        surface[:400,800:1200] = input_data['bev']
        # surface[:400,800:1200] = map
        # surface[400:600,800:1000] = input_data['map_t1']
        # surface[400:600,1000:1200] = input_data['map_t2']
        surface[:150,:200] = input_data['rgb_left']
        surface[:150, 600:800] = input_data['rgb_right']
        surface[:150, 325:475] = input_data['rgb_focus']
        surface = cv2.putText(surface, input_data['control'], (20,580), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255), 1)
        surface = cv2.putText(surface, input_data['speed'], (20,560), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255), 1)
        # surface = cv2.putText(surface, input_data['meta_infos'][0], (20,560), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255), 1)
        # surface = cv2.putText(surface, input_data['meta_infos'][1], (20,540), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255), 1)
        surface = cv2.putText(surface, input_data['time'], (20,520), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255), 1)

        surface = cv2.putText(surface, 'Left  View', (40,135), cv2.FONT_HERSHEY_SIMPLEX,0.75,(0,0,0), 2)
        surface = cv2.putText(surface, 'Focus View', (335,135), cv2.FONT_HERSHEY_SIMPLEX,0.75,(0,0,0), 2)
        surface = cv2.putText(surface, 'Right View', (640,135), cv2.FONT_HERSHEY_SIMPLEX,0.75,(0,0,0), 2)

        surface = cv2.putText(surface, 'Future Prediction', (940,420), cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0), 2)
        surface = cv2.putText(surface, 't', (1160,385), cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,0), 2)
        surface = cv2.putText(surface, '0', (1170,385), cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0), 2)
        surface = cv2.putText(surface, 't', (960,585), cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,0), 2)
        surface = cv2.putText(surface, '1', (970,585), cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0), 2)
        surface = cv2.putText(surface, 't', (1160,585), cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,0), 2)
        surface = cv2.putText(surface, '2', (1170,585), cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0), 2)

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
        self.lidar_processed = list()
        self.track = autonomous_agent.Track.SENSORS
        self.step = -1
        self.wall_start = time.time()
        self.initialized = False
        self.rgb_front_transform = create_carla_rgb_transform(224)
        self.rgb_left_transform = create_carla_rgb_transform(128)
        self.rgb_right_transform = create_carla_rgb_transform(128)
        self.rgb_center_transform = create_carla_rgb_transform(128, need_scale=False)

        self.tracker = Tracker()

        # self.input_buffer = {
        #     "rgb": deque(),
        #     "rgb_left": deque(),
        #     "rgb_right": deque(),
        #     "rgb_rear": deque(),
        #     "lidar": deque(),
        #     "gps": deque(),
        #     "thetas": deque(),
        # }
        
        # self._vehicle = CarlaDataProvider.get_hero_actor()
        # self._world = self._vehicle.get_world()
        # self.birdview_producer = BirdViewProducer(
        #     CarlaDataProvider.get_client(),  # carla.Client
        #     target_size=PixelDimensions(width=400, height=400),
        #     pixels_per_meter=4,
        #     crop_type=BirdViewCropType.FRONT_AND_REAR_AREA,
        # )

        self.config = imp.load_source("MainModel", path_to_conf_file).GlobalConfig(setting='eval')
        # self.config = GlobalConfig(setting='eval')
        self.skip_frames = self.config.skip_frames
        self.controller = InterfuserController(self.config)
        self.seq_len = self.config.seq_len
        self.img_states = np.zeros(shape=(self.config.seq_len, 3,
                                          self.config.img_resolution[0],
                                          self.config.img_resolution[1])
                                   )
        # print("image states shape:", self.img_states.shape)

        self.decision_states = np.zeros(shape=(self.seq_len, len(RoadOption) - 1))
        self.decision_states[:, 3] = 1.0
        self.detection_states = np.zeros(shape=(self.seq_len, self.config.state_dim - len(RoadOption) + 1))
        # print("decision states shape", self.decision_states.shape, "detection states shape", self.detection_states.shape)
        
        self.actions = np.zeros(shape=(self.seq_len, self.config.num_classes))
        self.rtgs = np.ones(shape=(self.seq_len, 1)) * 901
        self.timesteps = np.zeros(shape=(self.seq_len, 1))
        # print("action shape:", self.actions.shape, "rtg shape:", self.rtgs.shape, "timestep shape:", self.timesteps.shape)

        ######### Load the trained model #########
        # Create model and optimizers
        # prompt_param = [self.config.num_tasks, [self.config.pool_size, self.config.e_prompt_len,
        #                                         self.config.g_prompt_len, self.config.top_k]]
        # self.net = vit_pt_imnet(out_dim=self.config.num_classes, prompt_flag=self.config.prompt_name,
        #                      prompt_param=prompt_param)

        self.net = vit_pt_imnet(self.config)
        path_to_model_file = self.config.model_path
        self.net.load_state_dict(torch.load(path_to_model_file)["model"])
        self.net.cuda(device=self.device)
        self.net.eval()
        
        if isinstance(self.config.model_path, list):
            self.ensemble = True
        else:
            self.ensemble = False

        # if self.ensemble:
        #     for i in range(len(self.config.model)):
        #         self.nets = []
        #         net = create_model(self.config.model[i])
        #         path_to_model_file = self.config.model_path[i]
        #         print('load model: %s' % path_to_model_file)
        #         net.load_state_dict(torch.load(path_to_model_file)["state_dict"])
        #         net.cuda()
        #         net.eval()
        #         self.nets.append(net)
        # else:
        #     self.net = create_model(self.config.model)
        #     path_to_model_file = self.config.model_path
        #     print('load model: %s' % path_to_model_file)
        #     self.net.load_state_dict(torch.load(path_to_model_file)["state_dict"])
        #     self.net.cuda()
        #     self.net.eval()

        self.softmax = torch.nn.Softmax(dim=1)
        self.traffic_meta_moving_avg = np.zeros((400, 7))
        self.momentum = self.config.momentum
        self.prev_lidar = None
        self.prev_control = None
        self.prev_action = np.zeros(2)
        self.prev_surround_map = None

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

            print(string)

            self.save_path = pathlib.Path(SAVE_PATH) / string
            self.save_path.mkdir(parents=True, exist_ok=False)
            (self.save_path / "meta").mkdir(parents=True, exist_ok=False)

    def _init(self):
        self._route_planner = RoutePlanner(4.0, 50.0)
        self._route_planner.set_route(self._global_plan, True)
        
        self._command_planner = RoutePlanner(7.5, 25.0, 257)
        self._command_planner.set_route(self._global_plan, True)
        
        self.initialized = True
        self._hic = DisplayInterface()
        self._vehicle = CarlaDataProvider.get_hero_actor()
        self._world = self._vehicle.get_world()
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

        result["gps"] = pos
        
        #######################################################################
        next_wp, next_cmd = self._route_planner.run_step(pos)
        _, _ = self._command_planner.run_step(gps)
        #######################################################################
        
        result["next_command"] = next_cmd.value
        result['measurements'] = [pos[0], pos[1], compass, speed]

        theta = compass + np.pi / 2
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

        local_command_point = np.array([next_wp[0] - pos[0], next_wp[1] - pos[1]])
        local_command_point = R.T.dot(local_command_point)
        result["target_point"] = local_command_point

        return result

    @torch.no_grad()
    def run_step(self, input_data, timestamp):
        if not self.initialized:
            self._init()

        self.step += 1
        if self.step % self.skip_frames != 0 and self.step > 4:
            return self.prev_control

        tick_data = self.tick(input_data)

        velocity = tick_data["speed"]
        
        #######################################################################
        # command = tick_data["next_command"]
        command = self._command_planner.route[0][1].value
        #######################################################################

        rgb = (
            self.rgb_front_transform(Image.fromarray(tick_data["rgb"]))
            .unsqueeze(0)
            # .cuda()
            .float()
        )
        rgb_left = (
            self.rgb_left_transform(Image.fromarray(tick_data["rgb_left"]))
            .unsqueeze(0)
            # .cuda()
            .float()
        )
        rgb_right = (
            self.rgb_right_transform(Image.fromarray(tick_data["rgb_right"]))
            .unsqueeze(0)
            # .cuda()
            .float()
        )
        rgb_center = (
            self.rgb_center_transform(Image.fromarray(tick_data["rgb"]))
            .unsqueeze(0)
            # .cuda()
            .float()
        )
        
        img_states = torch.tensor(self.img_states, dtype=torch.float64).to(device=self.device)
        img_states = img_states / 255.
        decision_states = torch.tensor(self.decision_states, dtype=torch.float64).to(device=self.device)
        detection_states = torch.tensor(self.detection_states, dtype=torch.float64).to(device=self.device)
        actions = torch.tensor(self.actions, dtype=torch.float64).to(device=self.device)
        rtgs = torch.tensor(self.rtgs, dtype=torch.float64).to(device=self.device)
        timesteps = torch.tensor(self.timesteps, dtype=torch.int32).to(device=self.device)
        
        print(detection_states, 'and ', decision_states)
        
        start_time = time.time()

        if self.ensemble:
            outputs = []
            with torch.no_grad():
                for net in self.nets:
                    action = net(img_states, actions, rtgs, timesteps, detection_states,
                                 decision_states)
                    outputs.append(action)
            # traffic_meta = torch.mean(torch.stack([x[0] for x in outputs]), 0)
            # pred_waypoints = torch.mean(torch.stack([x[1] for x in outputs]), 0)
            # is_junction = torch.mean(torch.stack([x[2] for x in outputs]), 0)
            # traffic_light_state = torch.mean(torch.stack([x[3] for x in outputs]), 0)
            # stop_sign = torch.mean(torch.stack([x[4] for x in outputs]), 0)
            # bev_feature = torch.mean(torch.stack([x[5] for x in outputs]), 0)
        else:
            with torch.no_grad():
                # (
                #     traffic_meta,
                #     pred_waypoints,
                #     is_junction,
                #     traffic_light_state,
                #     stop_sign,
                #     bev_feature,
                # ) = self.net(input_data)
                action = self.net(img_states, actions, rtgs, timesteps, detection_states,
                                  decision_states)

        action = action.squeeze(0)[-1,:].detach().cpu().numpy()

        total_time = time.time() - start_time
        # total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        # print(f"Total training time: {total_time}")

        # traffic_meta = traffic_meta.detach().cpu().numpy()[0]
        # bev_feature = bev_feature.detach().cpu().numpy()[0]
        # pred_waypoints = pred_waypoints.detach().cpu().numpy()[0]
        # is_junction = self.softmax(is_junction).detach().cpu().numpy().reshape(-1)[0]
        # traffic_light_state = (
        #     self.softmax(traffic_light_state).detach().cpu().numpy().reshape(-1)[0]
        # )
        # stop_sign = self.softmax(stop_sign).detach().cpu().numpy().reshape(-1)[0]


        # if self.step % 2 == 0 or self.step < 4:
        #     traffic_meta = self.tracker.update_and_predict(traffic_meta.reshape(20, 20, -1), tick_data['gps'], tick_data['compass'], self.step // 2)
        #     traffic_meta = traffic_meta.reshape(400, -1)
        #     self.traffic_meta_moving_avg = (
        #         self.momentum * self.traffic_meta_moving_avg
        #         + (1 - self.momentum) * traffic_meta
        #     )
        # traffic_meta = self.traffic_meta_moving_avg

        # tick_data["raw"] = traffic_meta
        # tick_data["bev_feature"] = bev_feature

        # steer, throttle, brake, meta_infos = self.controller.run_step(
        #     velocity,
        #     pred_waypoints,
        #     is_junction,
        #     traffic_light_state,
        #     stop_sign,
        #     self.traffic_meta_moving_avg,
        # )

        # if brake < 0.05:
        #     brake = 0.0
        # if brake > 0.1:
        #     throttle = 0.0

        steer = action[1]
        pedal = action[0]
        
        if pedal >= 0.0:
            throttle = pedal
            brake = 0.0
        else:
            throttle = 0.0
            brake = pedal
            
        if self.step < 4:
            print('init')
            throttle = 0.0
            brake = 0.0
        
        control = carla.VehicleControl()
        control.steer = float(steer)
        control.throttle = float(throttle)
        control.brake = float(brake)

        # surround_map, box_info = render(traffic_meta.reshape(20, 20, 7), pixels_per_meter=20)
        # surround_map = surround_map[:400, 160:560]
        # surround_map = np.stack([surround_map, surround_map, surround_map], 2)

        self_car_map = render_self_car(
            loc=np.array([0, 0]),
            ori=np.array([0, -1]),
            box=np.array([2.45, 1.0]),
            color=[1, 1, 0], pixels_per_meter=20
        )[:400, 160:560]

        # pred_waypoints = pred_waypoints.reshape(-1, 2)
        # safe_index = 10
        # for i in range(10):
        #     if pred_waypoints[i, 0] ** 2 + pred_waypoints[i, 1] ** 2> (meta_infos[3]+0.5) ** 2:
        #         safe_index = i
        #         break
        # wp1 = render_waypoints(pred_waypoints[:safe_index], pixels_per_meter=20, color=(0, 255, 0))[:400, 160:560]
        # wp2 = render_waypoints(pred_waypoints[safe_index:], pixels_per_meter=20, color=(255, 0, 0))[:400, 160:560]
        # wp = wp1 + wp2

        # surround_map = np.clip(
        #     (
        #         surround_map.astype(np.float32)
        #         + self_car_map.astype(np.float32)
        #         + wp.astype(np.float32)
        #     ),
        #     0,
        #     255,
        # ).astype(np.uint8)

        # map_t1, box_info = render(traffic_meta.reshape(20, 20, 7), pixels_per_meter=20, t=1)
        # map_t1 = map_t1[:400, 160:560]
        # map_t1 = np.stack([map_t1, map_t1, map_t1], 2)
        # map_t1 = np.clip(map_t1.astype(np.float32) + self_car_map.astype(np.float32), 0, 255).astype(np.uint8)
        # map_t1 = cv2.resize(map_t1, (200, 200))
        # map_t2, box_info = render(traffic_meta.reshape(20, 20, 7), pixels_per_meter=20, t=2)
        # map_t2 = map_t2[:400, 160:560]
        # map_t2 = np.stack([map_t2, map_t2, map_t2], 2)
        # map_t2 = np.clip(map_t2.astype(np.float32) + self_car_map.astype(np.float32), 0, 255).astype(np.uint8)
        # map_t2 = cv2.resize(map_t2, (200, 200))


        cmd_one_hot = [0, 0, 0, 0, 0, 0]
        cmd = command - 1
        cmd_one_hot[cmd] = 1
        # cmd_one_hot.append(velocity)
        decision = np.array(cmd_one_hot)
        # print("command:", command, "decision:", decision)
        # print(rgb.shape)

        # input_data = {}
        # input_data["rgb"] = rgb
        # input_data["rgb_left"] = rgb_left
        # input_data["rgb_right"] = rgb_right
        # input_data["rgb_center"] = rgb_center
        # input_data["measurements"] = mes
        # input_data["target_point"] = (
        #     torch.from_numpy(tick_data["target_point"]).float().cuda().view(1, -1)
        # )
        # input_data["lidar"] = (
        #     torch.from_numpy(tick_data["lidar"]).float().cuda().unsqueeze(0)
        # )
        
        # detections
        actors = self._world.get_actors()

        light = self._is_light_red(actors.filter("*traffic_light*"))
        walker = self._is_walker_hazard(actors.filter("*walker*"))
        bike = self._is_bike_hazard(actors.filter("*vehicle*"))
        stop_sign = self._is_stop_sign_hazard(actors.filter("*stop*"))

        # record the reason for braking
        bike = [x.id for x in bike]
        pedestrian = [x.id for x in walker]
        red_light = [x.id for x in light]
        stop_sign = [x.id for x in stop_sign]
        
        full_list = [bike, pedestrian, red_light, stop_sign]
        detection = np.zeros((len(full_list)))
        for idx, item in enumerate(full_list):
            if len(item) > 0:
                target_speed = 0.0
                detection[idx] = 1.0
            else:
                target_speed = self.config.max_speed
        
        # print("bike:", bike, "ped:", pedestrian, "light:", red_light, "stop:", stop_sign, "det:", detection)
        
        action = np.array((pedal, steer))
        timestep = np.array(self.step)[np.newaxis]
        
        # reward calculation
        
        # action reward
        if abs(steer - self.prev_action[-1]) > 0.01:
            r_action = -0.1
        else:
            r_action = 0.0
        
        # speed reward
        r_speed = 1.0 - np.abs(velocity - target_speed) / self.config.max_speed

        # lateral displacement reward
        ego_theta = tick_data["compass"]
        pos = self._get_position(tick_data)
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
        
        self.states_adapter(rgb, decision, detection, action, reward, timestep)
        self.prev_action = action

        if self.step % 2 != 0 and self.step > 4:
            control = self.prev_control
        else:
            self.prev_control = control
            # self.prev_surround_map = surround_map

        # tick_data["map"] = self.prev_surround_map
        # tick_data["map_t1"] = map_t1
        # tick_data["map_t2"] = map_t2
        tick_data["rgb_raw"] = tick_data["rgb"]
        tick_data["rgb_left_raw"] = tick_data["rgb_left"]
        tick_data["rgb_right_raw"] = tick_data["rgb_right"]

        tick_data["rgb"] = cv2.resize(tick_data["rgb"], (800, 600))
        tick_data["rgb_left"] = cv2.resize(tick_data["rgb_left"], (200, 150))
        tick_data["rgb_right"] = cv2.resize(tick_data["rgb_right"], (200, 150))
        tick_data["rgb_focus"] = cv2.resize(tick_data["rgb_raw"][244:356, 344:456], (150, 150))
        tick_data["control"] = "throttle: %.2f, steer: %.2f, brake: %.2f" % (
            control.throttle,
            control.steer,
            control.brake,
        )
        tick_data["speed"] = "speed: %.2f, target_speed: %.2f" % (velocity, self.config.max_speed)
        bev = tick_data["raw_lidar"]
        bev = BirdViewProducer.as_rgb(
            self.birdview_producer.produce(agent_vehicle=self._vehicle)
        )
        tick_data["bev"] = cv2.resize(bev, (400, 400))
        # tick_data["meta_infos"] = meta_infos
        # tick_data["box_info"] = "car: %d, bike: %d, pedestrian: %d" % (
        #     box_info["car"],
        #     box_info["bike"],
        #     box_info["pedestrian"],
        # )
        tick_data["mes"] = "speed: %.2f" % velocity
        tick_data["time"] = "time: %.3f" % timestamp
        surface = self._hic.run_interface(tick_data)
        tick_data["surface"] = surface

        if SAVE_PATH is not None:
            self.save(tick_data)

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

    def states_adapter(self, img_states, decision_states, detection_states, actions,
                       rewards, timesteps):
        # print("before timesteps:", self.timesteps)
        self.img_states = np.concatenate((self.img_states, img_states), axis=0)[-self.seq_len:]
        self.decision_states = np.concatenate((self.decision_states, decision_states[np.newaxis]), axis=0)[-self.seq_len:]
        self.detection_states = np.concatenate((self.detection_states, detection_states[np.newaxis]), axis=0)[-self.seq_len:]
        self.actions = np.concatenate((self.actions, actions[np.newaxis]), axis=0)[-self.seq_len:]
        self.rtgs = np.concatenate((self.rtgs, (self.rtgs[-1] - rewards)[np.newaxis]), axis=0)[-self.seq_len:]
        self.timesteps = np.concatenate((self.timesteps, timesteps[np.newaxis]), axis=0)[-self.seq_len:]
        # print("after timesteps:", self.timesteps)
        
        # self.img_states[0:self.config.seq_len-1,:,:,:] = self.img_states[1:self.config.seq_len,:,:,:]
        # self.img_states[self.config.seq_len,:,:,:] = img_states
        # np.moveaxis(self.img_states, -1, 1)
        
        # self.decision_states[0:self.config.seq_len-1,:] = self.decision_states[1:self.config.seq_len,:]
        # self.decision_states[self.config.seq_len,:] = decision_states
    
        # self.detection_states[0:self.config.seq_len-1,:] = self.detection_states[1:self.config.seq_len,:]
        # self.detection_states[self.config.seq_len,:] = detection_states
        # return self.img_states, self.condition_states

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
                print('walker:', collides, collision_point)
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
                print('bike:', collides, collision_point)
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