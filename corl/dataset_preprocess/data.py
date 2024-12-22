#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 10:51:09 2023

@author: oscar
"""

import os
import ujson
from skimage.transform import rotate
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
import sys
from pathlib import Path
import cv2
import random
from copy import deepcopy
import io
import torch
import pandas as pd
from enum import Enum
import matplotlib.pyplot as plt

from corl.utils import get_vehicle_to_virtual_lidar_transform, get_vehicle_to_lidar_transform, get_lidar_to_vehicle_transform, get_lidar_to_bevimage_transform

from PIL import Image
# from foundation_model.vision_encoder.segformer import SegEncoder
# from foundation_model.MobileVLM.mask2former import SegEncoder

class RoadOption(Enum):
    """
    RoadOption represents the possible topological configurations when moving from a segment of lane to other.
    """
    # VOID = -1
    LEFT = 1
    RIGHT = 2
    STRAIGHT = 3
    LANEFOLLOW = 4
    CHANGELANELEFT = 5
    CHANGELANERIGHT = 6

class CARLA_Data(Dataset):

    def __init__(self, root, config, shared_dict=None):
        self.config = config
        self.seq_len = np.array(config.seq_len)
        self.pred_len = np.array(config.pred_len)

        self.img_resolution = np.array(config.img_resolution)
        self.img_width = np.array(config.img_width)
        self.scale = np.array(config.scale)
        self.data_cache = shared_dict
        self.augment = np.array(config.augment)
        self.aug_max_rotation = np.array(config.aug_max_rotation)
        self.inv_augment_prob = np.array(config.inv_augment_prob)
        # self.converter = np.uint8(config.converter)
                
        self.max_speed = 6.0 # m/s 
        self.reward_scale = 100

        self.images = []
        # self.bevs = []
        self.measurements = []
        self.done_idxs = []
        self.step_rewards = []
        self.throttles = []
        self.brakes = []
        self.steers = []
        self.local_decisions = []
        self.speed_decisions = []
        self.detections = []
        self.reference_paths = []
        self.ego_paths = []
        self.target_waypoints = []
        # self.target_velocities = []
        self.masks = []

        # self.segformer = SegEncoder(config.device)
        for sub_root in tqdm(root, file=sys.stdout):
            # sub_root = Path(sub_root + '/data/') # data folder path 
            sub_root = Path(sub_root) # data folder path 
            # list sub-directories in root
            root_files = os.listdir(sub_root) # index the data folder
            routes = [folder for folder in root_files if not os.path.isfile(os.path.join(sub_root,folder))]
            self.route_num = len(routes)
            for route in tqdm(routes):
                route_dir = sub_root / route
                # seg_folder = '/segmentation_front_HD/'
                # seg_dir = str(route_dir) + seg_folder
                # if (not os.path.isdir(seg_dir)):
                #     print('Created dir:', seg_dir)
                #     os.makedirs(seg_dir, exist_ok=True)
                if config.image_type == 'segmentation':
                    prefix = "segmentation_front_HD"
                else:
                    prefix = "rgb_front"
                    
                num_seq = len(os.listdir(route_dir / prefix))
                skip_counter = 0
                
                images = []
                measurements = []
                done_idxs = []
                step_rewards = []
                throttles = []
                brakes = []
                steers = []
                local_decisions = []
                speed_decisions = []
                detections = []
                reference_paths = []
                ego_paths = []
                target_waypoints = []
                masks = []                
                
                for seq in range(1, num_seq): # skip first frame
                    skip_flag = False

                    image_i = np.array(route_dir / prefix / ("%04d.jpg" % (seq))).astype(np.string_)
                    measurement_i = np.array(route_dir / "measurements" / ("%04d.json"%(seq))).astype(np.string_)

                    # open the data
                    with open(str(measurement_i.astype(np.string_), encoding='utf-8'), 'r') as f1:
                        measurement_i = ujson.load(f1)
                    
                    for i in measurement_i:
                        if np.any(pd.isna(measurement_i[i])):
                            skip_flag = True
                            print('skip the {}th frame of route {}, because {} has nan'.format(seq, route, i))
                            
                    if skip_flag:
                        skip_counter += 1
                        continue
                    
                    image_i = cv2.imread(str(image_i, encoding='utf-8'), cv2.IMREAD_COLOR)
                    if(image_i is None):
                        print("Error loading file: ", str(image_i, encoding='utf-8'))
                    image_i = scale_image_cv2(cv2.cvtColor(image_i, cv2.COLOR_BGR2RGB), self.scale)
                    image_i = cv2.resize(image_i, tuple(self.img_resolution))
                    
                    # segmentation_image = self.segformer.forward(image_i)
                    # cv2.imwrite(os.path.join(str(route_dir) + seg_folder, ('%04d.jpg' % (seq))), segmentation_image)
                    # # im = Image.fromarray(segmentation_image)
                    # # im.save(os.path.join(str(route_dir) + seg_folder, ('%04d.jpg' % (seq))))
                    # del segmentation_image
                    # continue
                    image_i = np.transpose(image_i, (2, 0, 1))

                    bike = measurement_i['is_bike_present']
                    pedestrian = measurement_i['is_pedestrian_present']
                    red_light = measurement_i['is_red_light_present']
                    stop_sign = measurement_i['is_stop_sign_present']
                    vehicle = measurement_i['is_vehicle_present']
                    junction_vehicle = measurement_i['is_junction_vehicle_present']
                    vehicle = vehicle + junction_vehicle
                    full_list = [vehicle, bike, pedestrian, red_light, stop_sign]
                    
                    detection = np.zeros((len(full_list)))
                    for idx, item in enumerate(full_list):
                        if len(item) > 0:
                            detection[idx] = 1.0

                    local_decision = np.zeros(len(RoadOption))
                    local_decision[measurement_i['command']-1] = 1.0
                    
                    slow = measurement_i['should_slow']
                    brake = measurement_i['should_brake']
                    
                    if brake:
                        speed_decision = np.array([0., 0., 1.])
                    elif slow:
                        speed_decision = np.array([0., 1., 0.])
                    else:
                        speed_decision = np.array([1., 0., 0.])

                    mask = 1

                    ego_x = measurement_i['gps_x']
                    ego_y = measurement_i['gps_y']
                    ego_theta = measurement_i['theta']
                    target_x = measurement_i['far_node_x']
                    target_y = measurement_i['far_node_y']
                    future_waypoints = measurement_i['future_waypoints'][0:self.pred_len]
                                        
                    R = np.array([
                        [np.cos(np.pi/2+ego_theta), -np.sin(np.pi/2+ego_theta)],
                        [np.sin(np.pi/2+ego_theta),  np.cos(np.pi/2+ego_theta)]
                        ])

                    target_waypoint = np.array([target_x-ego_x, target_y-ego_y])
                    target_waypoint = R.T.dot(target_waypoint)

                    ego_pos = np.array([ego_x, ego_y, ego_theta])
                    
                    reference_path = []
                    for i in range(self.pred_len):
                        j = min(i, len(future_waypoints)-1)
                        next_wp = future_waypoints[j][:2]
                        next_local_wp = R.T.dot(next_wp - np.array((ego_x, ego_y)))
                        reference_path.append(next_local_wp)

                    images.append(image_i)
                    measurements.append(measurement_i)
                    steers.append(measurement_i['steer'])
                    throttles.append(measurement_i['throttle'])
                    brakes.append(measurement_i['brake'])
                    step_rewards.append(measurement_i['reward'])
                    detections.append(detection)
                    local_decisions.append(local_decision)
                    speed_decisions.append(speed_decision)
                    target_waypoints.append(target_waypoint)
                    reference_paths.append(np.array(reference_path))
                    ego_paths.append(ego_pos)
                    masks.append(mask)
                
                seudo_image = np.zeros(shape=(3, 3,
                                              config.img_resolution[1],
                                              config.img_resolution[0]),
                                            dtype=np.int32,
                                            )
                
                seudo_step_rewards = np.zeros(shape=(3)) #* self.step_rewards[0]
                seudo_throttles = np.zeros(shape=(3)) #* self.throttles[0]
                seudo_brakes = np.zeros(shape=(3)) #* self.brakes[0]
                seudo_steers = np.zeros(shape=(3)) #* self.steers[0]
                seudo_local_decisions = np.ones(shape=(3, len(local_decisions[0]))) * local_decisions[0]
                seudo_speed_decisions = np.ones(shape=(3, len(speed_decisions[0]))) * speed_decisions[0]
                seudo_detections = np.ones(shape=(3, len(detections[0]))) * detections[0]
                seudo_target_waypoints = np.ones(shape=(3, len(target_waypoints[0]))) * target_waypoints[0]
                seudo_masks = np.zeros(shape=(3))
                seudo_reference_path = np.ones(shape=(len(reference_paths[0]), len(reference_paths[0][0]))) * reference_paths[0] 
                seudo_ego_path = np.ones(shape=(3, len(ego_paths[0]))) * ego_paths[0]
                
                images = np.insert(images, 0, seudo_image, axis=0).tolist()
                step_rewards = np.insert(step_rewards, 0, seudo_step_rewards, axis=0).tolist()
                throttles = np.insert(throttles, 0, seudo_throttles, axis=0).tolist()
                brakes = np.insert(brakes, 0, seudo_brakes, axis=0).tolist()
                steers = np.insert(steers, 0, seudo_steers, axis=0).tolist()
                local_decisions = np.insert(local_decisions, 0, seudo_local_decisions, axis=0).tolist()
                speed_decisions = np.insert(speed_decisions, 0, seudo_speed_decisions, axis=0).tolist()
                detections = np.insert(detections, 0, seudo_detections, axis=0).tolist()
                target_waypoints = np.insert(target_waypoints, 0, seudo_target_waypoints, axis=0).tolist()
                masks = np.insert(masks, 0, seudo_masks, axis=0).tolist()

                for i in range(3):
                    reference_paths.insert(0, seudo_reference_path)
                ego_paths = np.insert(ego_paths, 0, seudo_ego_path, axis=0).tolist()
                done_idxs.append(num_seq - 1 - skip_counter + 3) # skip first frame

                ### process the entire ego path to a number of trajectories
                window_size = self.pred_len
                num_windows = len(ego_paths)
                        
                ego_path = []
                shift = 1 #2
                for window in range(num_windows):
                    ego_traj = []
                    
                    ego_theta = ego_paths[window][-1]
                    ego_waypoint = ego_paths[window][:2]
                    
                    R = np.array([
                        [np.cos(np.pi/2+ego_theta), -np.sin(np.pi/2+ego_theta)],
                        [np.sin(np.pi/2+ego_theta),  np.cos(np.pi/2+ego_theta)]
                        ])
                    
                    for i in range(window_size):
                        
                        j = min(num_windows-1, window + shift*(i+1))
                        waypoint = np.array(ego_paths[j])[:2]
                        
                        local_waypoint = R.T.dot(np.array(waypoint) - np.array(ego_waypoint))
                        ego_traj.append(local_waypoint)
                        
                    ego_path.append(np.array(ego_traj))

                self.images.append(np.array(images))
                self.step_rewards.append(np.array(step_rewards))
                self.throttles.append(np.array(throttles))
                self.brakes.append(np.array(brakes))
                self.steers.append(np.array(steers))
                self.local_decisions.append(np.array(local_decisions))
                self.speed_decisions.append(np.array(speed_decisions))
                self.detections.append(np.array(detections))
                self.target_waypoints.append(np.array(target_waypoints))
                self.reference_paths.append(np.array(reference_paths))
                self.ego_paths.append(np.array(ego_path))
                self.done_idxs.append(np.array(done_idxs))
                self.masks.append(np.array(masks))

        self.images = np.concatenate(self.images)
        self.step_rewards = np.concatenate(self.step_rewards)
        self.throttles = np.concatenate(self.throttles)
        self.brakes = np.concatenate(self.brakes)
        self.steers = np.concatenate(self.steers)
        self.local_decisions = np.concatenate(self.local_decisions)
        self.speed_decisions = np.concatenate(self.speed_decisions)
        self.detections = np.concatenate(self.detections)
        self.target_waypoints = np.concatenate(self.target_waypoints)
        self.reference_paths = np.concatenate(self.reference_paths)
        self.ego_paths = np.concatenate(self.ego_paths)
        self.done_idxs = np.concatenate(self.done_idxs)
        self.masks = np.concatenate(self.masks)

        # -- create reward-to-go dataset
        start_index = 0
        self.rtgs = np.zeros_like(self.step_rewards)
        for i in self.done_idxs:
            i = int(i + start_index)
            curr_traj_returns = self.step_rewards[start_index:i]
            for j in range(i-1, start_index-1, -1): # start from i-1
                rtg_j = curr_traj_returns[j-start_index:i-start_index]
                self.rtgs[j] = sum(rtg_j)
            print('Traj rtg is:', self.rtgs[j])
            start_index = i
        self.rtgs = self.rtgs / self.reward_scale
        print('\n max rtg is %f' % max(self.rtgs))

        # -- create timestep dataset
        start_index = 0
        self.timesteps = np.zeros(len(self.images), dtype=int)
        for i in self.done_idxs:
            i = int(i + start_index)
            self.timesteps[start_index:i] = np.arange(i - start_index)
            start_index = i
        print('\n max timestep is %d' % max(self.timesteps))
    
        self.rtgs = np.array(self.rtgs)    
        self.timesteps = np.array(self.timesteps)

    def __len__(self):
        """Returns the length of the dataset. """
        return self.images.shape[0] #- self.seq_len

    def __getitem__(self, index):
        """Returns the item at index idx. """
        cv2.setNumThreads(0) # Disable threading because the data loader will already split in threads.
        # backbone = str(self.backbone, encoding='utf-8')
        
        data = dict()
        
        done_idx = index + self.seq_len
        cum_idx = -1
        for i in self.done_idxs:
            cum_idx += i
            if cum_idx >= index: 
                done_idx = min(int(cum_idx), done_idx) #redefine the done_idx
                break
        idx = done_idx - self.seq_len
        # if done_idx >= 3542:
        #     print(self.done_idxs, index, cum_idx, idx, done_idx)
        
        states = torch.tensor(np.array(self.images[idx:done_idx]), dtype=torch.float32) #.reshape(self.seq_len, -1) # (block_size, 4*256*256)
        if self.config.model == 'GPT':
            states = states / 255.
        rtgs = torch.tensor(self.rtgs[idx:done_idx], dtype=torch.float32).unsqueeze(1)
        timesteps = torch.tensor(self.timesteps[idx:done_idx], dtype=torch.int32).unsqueeze(1)
        detections = torch.tensor(self.detections[idx:done_idx], dtype=torch.float32)
        route_decisions = torch.tensor(self.local_decisions[idx:done_idx], dtype=torch.float32)
        speed_decisions = torch.tensor(self.speed_decisions[idx:done_idx], dtype=torch.float32)
        target_waypoints = torch.tensor(self.target_waypoints[idx:done_idx], dtype=torch.float32)        
        masks = torch.tensor(self.masks[idx:done_idx], dtype=torch.int32)

        throttles = self.throttles[idx:done_idx]
        brakes = self.brakes[idx:done_idx]
        steers = self.steers[idx:done_idx]

        reference_path = torch.tensor(self.reference_paths[done_idx])    
        traj_len = reference_path.shape[0]
        reference_path = torch.cat([reference_path, torch.ones((self.pred_len - traj_len, reference_path.shape[1])) * reference_path[-1]], dim=0)

        ego_path = torch.tensor(self.ego_paths[done_idx])
        traj_len = ego_path.shape[0]
        ego_path = torch.cat([ego_path, torch.ones((self.pred_len - traj_len, ego_path.shape[1])) * ego_path[-1]], dim=0)

        # if torch.max(ego_path) > 30 or torch.min(ego_path) < -30:
        #     print('wtf')

        pedals = np.ones_like(throttles)
        pedals[brakes == True] = -1
        pedals[brakes == False] = throttles[brakes == False]
        actions = torch.tensor(np.stack((pedals, steers), axis=1),) # (block_size, 1)
        
        # data['local_waypoints'] = command_waypoints
        data['states'] = states
        data['actions'] = actions
        data['rtgs'] = rtgs
        data['timesteps'] = timesteps
        data['detections'] = detections
        data['route_decisions'] = route_decisions
        data['speed_decisions'] = speed_decisions
        data['target_waypoints'] = target_waypoints
        data['reference_path'] = reference_path
        data['ego_path'] = ego_path
        data['masks'] = masks
        return data

def get_depth(data):
    """
    Computes the normalized depth
    """
    data = np.transpose(data, (1,2,0))
    data = data.astype(np.float64)

    normalized = np.dot(data, [65536.0, 256.0, 1.0]) 
    normalized /=  (256 * 256 * 256 - 1)
    # in_meters = 1000 * normalized
    #clip to 50 meters
    normalized = np.clip(normalized, a_min=0.0, a_max=0.05)
    normalized = normalized * 20.0 # Rescale map to lie in [0,1]

    return normalized


def get_waypoints(labels, len_labels):
    assert(len(labels) == len_labels)
    num = len_labels
    waypoints = {}
    
    for result in labels[0]:
        car_id = result["id"]
        waypoints[car_id] = [[result['ego_matrix'], True]]
        for i in range(1, num):
            for to_match in labels[i]:
                if to_match["id"] == car_id:
                    waypoints[car_id].append([to_match["ego_matrix"], True])

    Identity = list(list(row) for row in np.eye(4))
    # padding here
    for k in waypoints.keys():
        while len(waypoints[k]) < num:
            waypoints[k].append([Identity, False])
    return waypoints

def align(lidar_0, measurements_0, measurements_1, degree=0):
    
    matrix_0 = measurements_0['ego_matrix']
    matrix_1 = measurements_1['ego_matrix']

    matrix_0 = np.array(matrix_0)
    matrix_1 = np.array(matrix_1)
   
    Tr_lidar_to_vehicle = get_lidar_to_vehicle_transform()
    Tr_vehicle_to_lidar = get_vehicle_to_lidar_transform()

    transform_0_to_1 = Tr_vehicle_to_lidar @ np.linalg.inv(matrix_1) @ matrix_0 @ Tr_lidar_to_vehicle

    # augmentation
    rad = np.deg2rad(degree)
    degree_matrix = np.array([[np.cos(rad), np.sin(rad), 0, 0],
                              [-np.sin(rad), np.cos(rad), 0, 0],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]])
    transform_0_to_1 = degree_matrix @ transform_0_to_1
                            
    lidar = lidar_0.copy()
    lidar[:, -1] = 1.
    #important we should convert the points back to carla format because when we save the data we negatived y component
    # and now we change it back 
    lidar[:, 1] *= -1.
    lidar = transform_0_to_1 @ lidar.T
    lidar = lidar.T
    lidar[:, -1] = lidar_0[:, -1]
    # and we change back here
    lidar[:, 1] *= -1.

    return lidar


def lidar_to_histogram_features(lidar):
    """
    Convert LiDAR point cloud into 2-bin histogram over 256x256 grid
    """
    def splat_points(point_cloud):
        # 256 x 256 grid
        pixels_per_meter = 8
        hist_max_per_pixel = 5
        x_meters_max = 16
        y_meters_max = 32
        xbins = np.linspace(-x_meters_max, x_meters_max, 32*pixels_per_meter+1)
        ybins = np.linspace(-y_meters_max, 0, 32*pixels_per_meter+1)
        hist = np.histogramdd(point_cloud[..., :2], bins=(xbins, ybins))[0]
        hist[hist>hist_max_per_pixel] = hist_max_per_pixel
        overhead_splat = hist/hist_max_per_pixel
        return overhead_splat

    below = lidar[lidar[...,2]<=-2.3]
    above = lidar[lidar[...,2]>-2.3]
    below_features = splat_points(below)
    above_features = splat_points(above)
    features = np.stack([above_features, below_features], axis=-1)
    features = np.transpose(features, (2, 0, 1)).astype(np.float64)
    features = np.rot90(features, -1, axes=(1,2)).copy()
    return features

def get_bbox_label(bbox, rad=0):
    # dx, dy, dz, x, y, z, yaw
    # ignore z
    dz, dx, dy, x, y, z, yaw, speed, brake =  bbox

    pixels_per_meter = 8

    # augmentation
    degree_matrix = np.array([[np.cos(rad), np.sin(rad), 0],
                              [-np.sin(rad), np.cos(rad), 0],
                              [0, 0, 1]])
    T = get_lidar_to_bevimage_transform() @ degree_matrix
    position = np.array([x, y, 1.0]).reshape([3, 1])
    position = T @ position

    position = np.clip(position, 0., 255.)
    x, y = position[:2, 0]
    # center_x, center_y, w, h, yaw
    bbox = np.array([x, y, dy*pixels_per_meter, dx*pixels_per_meter, 0, 0, 0])
    bbox[4] = yaw + rad
    bbox[5] = speed
    bbox[6] = brake
    return bbox


def parse_labels(labels, rad=0):
    bboxes = {}
    for result in labels:
        num_points = result['num_points']
        distance = result['distance']

        x = result['position'][0]
        y = result['position'][1]

        bbox = result['extent'] + result['position'] + [result['yaw'], result['speed'], result['brake']]
        bbox = get_bbox_label(bbox, rad)

        # Filter bb that are outside of the LiDAR after the random augmentation. The bounding box is now in image space
        if num_points <= 1 or bbox[0] <= 0.0 or bbox[0] >= 255.0 or bbox[1] <= 0.0 or bbox[1] >=255.0:
            continue

        bboxes[result['id']] = bbox
    return bboxes

def scale_image(image, scale):
    (width, height) = (int(image.width // scale), int(image.height // scale))
    im_resized = image.resize((width, height))
    return im_resized

def scale_image_cv2(image, scale):
    (width, height) = (int(image.shape[1] // scale), int(image.shape[0] // scale))
    im_resized = cv2.resize(image, (width, height))
    return im_resized

def crop_image(image, crop=(128, 640), crop_shift=0):
    """
    Scale and crop a PIL image, returning a channels-first numpy array.
    """
    width = image.width
    height = image.height
    crop_h, crop_w = crop
    start_y = height//2 - crop_h//2
    start_x = width//2 - crop_w//2
    
    # only shift for x direction
    start_x += int(crop_shift)

    image = np.asarray(image)
    cropped_image = image[start_y:start_y+crop_h, start_x:start_x+crop_w]
    cropped_image = np.transpose(cropped_image, (2,0,1))
    return cropped_image


def crop_image_cv2(image, crop=(128, 640), crop_shift=0):
    """
    Scale and crop a PIL image, returning a channels-first numpy array.
    """
    width = image.shape[1]
    height = image.shape[0]
    crop_h, crop_w = crop
    start_y = height // 2 - crop_h // 2
    start_x = width // 2 - crop_w // 2

    # only shift for x direction
    start_x += int(crop_shift)

    cropped_image = image[start_y:start_y + crop_h, start_x:start_x + crop_w]
    cropped_image = np.transpose(cropped_image, (2, 0, 1))
    return cropped_image

def scale_seg(image, scale):
    (width, height) = (int(image.shape[1] / scale), int(image.shape[0] / scale))
    if scale != 1:
        im_resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_NEAREST)
    else:
        im_resized = image
    return im_resized

def crop_seg(image, crop=(128, 640), crop_shift=0):
    """
    Scale and crop a seg image, returning a channels-first numpy array.
    """
    width = image.shape[1]
    height = image.shape[0]
    crop_h, crop_w = crop

    start_y = height//2 - crop_h//2
    start_x = width//2 - crop_w//2
    # only shift for x direction
    start_x += int(crop_shift)

    cropped_image = image[start_y:start_y+crop_h, start_x:start_x+crop_w]
    return cropped_image

def load_crop_bev_npy(bev_array, degree):
    """
    Load and crop an Image.
    Crop depends on augmentation angle.
    """
    PIXELS_PER_METER_FOR_BEV = 5
    PIXLES = 40 * PIXELS_PER_METER_FOR_BEV
    start_x = 200 - PIXLES // 2
    start_y = 200 - PIXLES // 2

    # shift the center by 7 because the lidar is + 1.3 in x 
    bev_array = np.moveaxis(bev_array, 0, -1).astype(np.float64)
    # bev_shift = np.zeros_like(bev_array)
    # bev_shift[7:] = bev_array[:-7]

    # bev_shift = rotate(bev_shift, degree)
    # cropped_image = bev_shift[start_y:start_y+PIXLES, start_x:start_x+PIXLES]
    cropped_image = bev_array
    cropped_image = np.moveaxis(cropped_image, -1, 0)

    # we need to predict others so append 0 to the first channel
    cropped_image = np.concatenate((np.zeros_like(cropped_image[:1]), 
                                    cropped_image[:1],
                                    cropped_image[:1] + cropped_image[1:2]), axis=0)

    cropped_image = np.argmax(cropped_image, axis=0)
    
    return cropped_image


def draw_target_point(target_point, color = (255, 255, 255)):
    image = np.zeros((256, 256), dtype=np.uint8)
    target_point = target_point.copy()

    # convert to lidar coordinate
    target_point[1] += 1.3
    point = target_point * 8.
    point[1] *= -1
    point[1] = 256 - point[1] 
    point[0] += 128 
    point = point.astype(np.int32)
    point = np.clip(point, 0, 256)
    cv2.circle(image, tuple(point), radius=5, color=color, thickness=3)
    image = image.reshape(1, 256, 256)
    return image.astype(np.float) / 255.

def correspondences_at_one_scale(valid_bev_points, valid_cam_points, lidar_x, lidar_y, camera_x, camera_y, scale):
    """
    Compute projections between LiDAR BEV and image space
    """
    cam_to_bev_proj_locs = np.zeros((lidar_x, lidar_y, 5, 2))
    bev_to_cam_proj_locs = np.zeros((camera_x, camera_y, 5, 2))

    tmp_bev = np.empty((lidar_x, lidar_y, ), dtype=object)
    tmp_cam = np.empty((camera_x, camera_y, ), dtype=object)
    for i in range(lidar_x):
        for j in range(lidar_y):
            tmp_bev[i,j] = []

    for i in range(camera_x):
        for j in range(camera_y):
            tmp_cam[i, j] = []

    for i in range(valid_bev_points.shape[0]):
        tmp_bev[valid_bev_points[i][0]//scale, valid_bev_points[i][1]//scale].append(valid_cam_points[i]//scale)
        tmp_cam[valid_cam_points[i][0]//scale, valid_cam_points[i][1]//scale].append(valid_bev_points[i]//scale)

    for i in range(lidar_x):
        for j in range(lidar_y):
            cam_to_bev_points = tmp_bev[i,j]

            if len(cam_to_bev_points) > 5:
                cam_to_bev_proj_locs[i,j] = np.array(random.sample(cam_to_bev_points, 5))
            elif len(cam_to_bev_points) > 0:
                num_points = len(cam_to_bev_points)
                cam_to_bev_proj_locs[i,j,:num_points] = np.array(cam_to_bev_points)

    for i in range(camera_x):
        for j in range(camera_y):
            bev_to_cam_points = tmp_cam[i,j]

            if len(bev_to_cam_points) > 5:
                bev_to_cam_proj_locs[i,j] = np.array(random.sample(bev_to_cam_points, 5))
            elif len(bev_to_cam_points) > 0:
                num_points = len(bev_to_cam_points)
                bev_to_cam_proj_locs[i,j,:num_points] = np.array(bev_to_cam_points)

    return cam_to_bev_proj_locs, bev_to_cam_proj_locs

def lidar_bev_cam_correspondences(world, lidar_vis=None, image_vis=None, step=None, debug=False):
    """
    Convert LiDAR point cloud to camera co-ordinates

    world: Expects the point cloud from CARLA in the CARLA coordinate system: x left, y forward, z up (LiDAR rotated by 90 degree)
    lidar_vis: lidar prjected to BEV
    image_vis: RGB input image to the network
    step: current timestep
    debug: Whether to save the debug images. If false only world is required
    """

    pixels_per_meter = 8
    lidar_width      = 256
    lidar_height     = 256
    lidar_meters_x   = (lidar_width  / pixels_per_meter) / 2 # Divided by two because the LiDAR is in the center of the image
    lidar_meters_y   =  lidar_height / pixels_per_meter

    downscale_factor = 32

    img_width  = 352
    img_height = 160
    fov_width  = 60

    left_camera_rotation  = -60.0
    right_camera_rotation =  60.0

    fov_height = 2.0 * np.arctan((img_height / img_width) * np.tan(0.5 * np.radians(fov_width)))
    fov_height = np.rad2deg(fov_height)

    # Our pixels are squares so focal_x = focal_y
    focal_x = img_width  / (2.0 * np.tan(np.deg2rad(fov_width)  / 2.0))
    focal_y = img_height / (2.0 * np.tan(np.deg2rad(fov_height) / 2.0))

    cam_z   = 2.3
    lidar_z = 2.5

    # get valid points in 64x64 grid
    world[:, 0] *= -1  # flip x axis, so that the positive direction points towards right. new coordinate system: x right, y forward, z up
    lidar = world[abs(world[:,0])<lidar_meters_x] # 32m to the sides
    lidar = lidar[lidar[:,1]<lidar_meters_y] # 64m to the front
    lidar = lidar[lidar[:,1]>0] # 0m to the back

    # Translate Lidar cloud to the same coordinate system as the cameras (They only differ in height)
    lidar[..., 2] = lidar[..., 2] + (lidar_z - cam_z)

    # Make copies because we will rotate the new pointclouds
    lidar_for_left_camera  = deepcopy(lidar)
    lidar_for_right_camera = deepcopy(lidar)


    lidar_indices = np.arange(0, lidar.shape[0], 1)
    # Use a pinhole camera model to project the LiDAR points onto the camera image
    z = lidar[..., 1]
    x = ((focal_x * lidar[..., 0]) / z) + (img_width  / 2.0)
    y = ((focal_y * lidar[..., 2]) / z) + (img_height / 2.0)
    result_center = np.stack([x, y, lidar_indices], 1)

    # Remove points that are outside of the image
    result_center = result_center[np.logical_and(result_center[...,0] > 0, result_center[...,0] < img_width)]
    result_center = result_center[np.logical_and(result_center[...,1] > 0, result_center[...,1] < img_height)]

    result_center_shifted = result_center
    result_center_shifted[..., 0] = result_center_shifted[..., 0] + (img_width / 2.0)

    # Rotate the left camera to align with the axis for projection with a pinhole camera model
    theta = np.radians(left_camera_rotation)
    R = np.array([
        [np.cos(theta), -np.sin(theta), 0.0],
        [np.sin(theta),  np.cos(theta), 0.0],
        [0.0,            0.0,           1.0]
    ])
    lidar_for_left_camera = R.dot(lidar_for_left_camera.T).T

    # Use a pinhole camera model to project the LiDAR points onto the camera image
    z = lidar_for_left_camera[..., 1]
    x = ((focal_x * lidar_for_left_camera[..., 0]) / z) + (img_width  / 2.0)
    y = ((focal_y * lidar_for_left_camera[..., 2]) / z) + (img_height / 2.0)
    result_left = np.stack([x, y, lidar_indices], 1)

    # Remove points that are outside of the image
    result_left = result_left[np.logical_and(result_left[...,0] > 0, result_left[...,0] < img_width)]
    result_left = result_left[np.logical_and(result_left[...,1] > 0, result_left[...,1] < img_height)]

    # We only use half of the left image, so we cut the unneccessary points
    result_left_shifted        = result_left[result_left[...,0] >= (img_width/2.0)]
    result_left_shifted[...,0] = result_left_shifted[...,0] - (img_width/2.0)

    # Do the same for the right image
    theta = np.radians(right_camera_rotation)
    R = np.array([
        [np.cos(theta), -np.sin(theta), 0.0],
        [np.sin(theta),  np.cos(theta), 0.0],
        [0.0,            0.0,           1.0]
    ])
    lidar_for_right_camera = R.dot(lidar_for_right_camera.T).T

    # Use a pinhole camera model to project the LiDAR points onto the camera image
    z = lidar_for_right_camera[..., 1]
    x = ((focal_x * lidar_for_right_camera[..., 0]) / z) + (img_width / 2.0)
    y = ((focal_y * lidar_for_right_camera[..., 2]) / z) + (img_height / 2.0)
    result_right = np.stack([x, y, lidar_indices], 1)

    # Remove points that are outside of the image
    result_right = result_right[np.logical_and(result_right[..., 0] > 0, result_right[..., 0] < img_width)]
    result_right = result_right[np.logical_and(result_right[..., 1] > 0, result_right[..., 1] < img_height)]

    # We only use half of the left image, so we cut the unneccessary points
    result_right_shifted = result_right[result_right[...,0] < (img_width/2.0)] # Cut of right part, it's not used.
    result_right_shifted[...,0] = result_right_shifted[...,0] + (img_width/2.0) + img_width

    # Combine the three images into one
    results_total = np.concatenate((result_left_shifted, result_center_shifted, result_right_shifted), axis=0)

    if(debug == True):
        # Visualize LiDAR hits in image
        vis = np.zeros([img_height, 2 * img_width])
        vis_bev = np.zeros([lidar_height, lidar_width])
        vis_original_image = image_vis[0].detach().cpu().numpy()
        vis_original_image = np.transpose(vis_original_image, (1, 2, 0)) / 255.0
        vis_original_lidar = np.zeros([lidar_height, lidar_width])
        lidar_vis = lidar_vis.detach().cpu().numpy()
        vis_original_lidar[np.greater(lidar_vis[0,0], 0)] = 255
        vis_original_lidar[np.greater(lidar_vis[0,1], 0)] = 255


    valid_bev_points = []
    valid_cam_points = []
    for i in range(results_total.shape[0]):
        # Project the LiDAR point to BEV and save index of the BEV image pixel.
        lidar_index = int(results_total[i, 2])
        bev_x = int((lidar[lidar_index][0] + lidar_meters_x) * pixels_per_meter)
        # The network input images use a top left coordinate system, we need to convert the bottom left coordinates by inverting the y axis
        bev_y = (int(lidar[lidar_index][1] * pixels_per_meter) - (lidar_height-1)) * -1

        valid_bev_points.append([bev_x, bev_y])
        # Calculate index in the final image by rounding down
        img_x = int(results_total[i][0])
        # The network input images use a top left coordinate system, we need to convert the bottom left coordinates by inverting the y axis
        img_y = (int(results_total[i][1]) - (img_height - 1)) * -1
        valid_cam_points.append([img_x, img_y])


        if (debug == True):
            vis_original_image[img_y, img_x] = np.array([0.0,1.0,0.0])
            vis_bev[bev_y, bev_x] = 255 #Debug visualization
            vis[img_y, img_x] = 255

    if (debug == True):
        # NOTE add the paths you want the images to land in here before debugging
        from matplotlib import pyplot as plt
        plt.ion()
        plt.imshow(vis_bev)
        plt.savefig(r'/home/hiwi/save folder/Visualizations/2/bev_lidar_{}.png'.format(step), bbox_inches='tight')
        plt.close()
        plt.imshow(vis_original_image)
        plt.savefig(r'/home/hiwi/save folder/Visualizations/2/image_with_lidar_{}.png'.format(step), bbox_inches='tight')
        plt.close()
        plt.ioff()


    valid_bev_points = np.array(valid_bev_points)
    valid_cam_points = np.array(valid_cam_points)

    bev_points, cam_points = correspondences_at_one_scale(valid_bev_points, valid_cam_points,  (lidar_width // downscale_factor),
                                                          (lidar_height // downscale_factor), (img_width // downscale_factor) * 2,
                                                          (img_height // downscale_factor), downscale_factor)

    return bev_points, cam_points

def decode_pil_to_npy(img):
    """
    """
    (channels, width, height) = (15, img.shape[1], img.shape[2])

    bev_array = np.zeros([channels, width, height])

    for ix in range(5):
        bit_pos = 8-ix-1
        bev_array[[ix, ix+5, ix+5+5]] = (img & (1<<bit_pos)) >> bit_pos

    # hard coded to select
    return bev_array[10:12]
