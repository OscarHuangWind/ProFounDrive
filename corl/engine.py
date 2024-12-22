#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 16:53:15 2023

@author: oscar
"""

"""
Train and eval functions used in main.py
"""
import os
import sys
import math
import json
import wandb
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Iterable

import torch
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import OneCycleLR

import utils
from timm.utils import accuracy

os.environ['WANDB_MODE'] = 'online'

class Engine(object):
    """
    Engine that runs training.
    """

    def __init__(self, model, data_loader, args, config, device, class_mask=None, parallel=False, cur_epoch=0):
        self.cur_epoch = cur_epoch
        self.bestval_epoch = cur_epoch
        self.train_loss = []
        self.val_loss = []
        self.bestval = 1e10
        self.model = model
        self.data_loader = data_loader
        self.args = args
        self.config = config
        self.device = device
        self.class_mask = class_mask
        self.parallel = parallel
        self.vis_save_path = self.args.logdir + r'/visualizations'
        self.task_name = self.args.task_seq #['town03', 'town06', 'town07']
        self.save_threshold = 0.0

    def train_one_epoch(self, model: torch.nn.Module, data_loader: Iterable, scaler: torch.cuda.amp.GradScaler,
                        optimizer: torch.optim.Optimizer, lr_scheduler: torch.optim.lr_scheduler,
                        device: torch.device, epoch: int, max_norm: float = 0, set_training_mode=True,
                        task_id=-1, class_mask=None, args = None,):
        
        trajectory_criterion = torch.nn.MSELoss().to(self.device)
        decision_criterion = torch.nn.CrossEntropyLoss().to(self.device)
        
        decision_loss_sum = 0.0
        waypoint_loss_sum = 0.0
        for it, data in enumerate(data_loader):
            states = data['states'].to(device)
            rtgs = data['rtgs'].to(device)
            timesteps = data['timesteps'].to(device)
            detections = data['detections'].to(device)
            route_actions = data['route_decisions'].to(device)
            speed_actions = data['speed_decisions'].to(device)
            decisions = None
            target_waypoints = data['target_waypoints'].to(device)
            reference_path = data['reference_path'].to(device)
            ego_path = data['ego_path'].to(device)
            masks = data['masks'].to(device)
            
            if self.config.label == 'local_planning':
                trajectory = ego_path
            else:
                trajectory = reference_path

            if self.config.decision == 'dual':
                actions = torch.cat((route_actions, speed_actions), dim=-1)
            else:
                actions = route_actions

            # combine the batch dimension and seq_len dimension
            states = states.reshape(-1, 3, self.config.img_resolution[1],
                                    self.config.img_resolution[0]).type(torch.float32).contiguous()
            decision_logits, wp_logits, prompt_loss = model.forward(states, actions, rtgs, timesteps, detections,
                                                                        decisions, target_waypoints=target_waypoints,
                                                                        attention_mask=masks, task_id=task_id, train=True)

            if self.config.decision == 'dual':
                route_actions = route_actions.reshape(-1, self.config.num_lat_classes).type(torch.float32).contiguous()
                route_action_targets = torch.argmax(route_actions, dim=1)
                route_decision_logits = decision_logits[:,:,:self.config.num_lat_classes]
                route_decision_logits = route_decision_logits.reshape(-1, self.config.num_lat_classes).type(torch.float32).contiguous()
                route_decision_loss = decision_criterion(route_decision_logits, route_action_targets)

                speed_actions = speed_actions.reshape(-1, self.config.num_long_classes).type(torch.float32).contiguous()
                speed_action_targets = torch.argmax(speed_actions, dim=1)
                speed_decision_logits = decision_logits[:,:,self.config.num_lat_classes:]
                speed_decision_logits = speed_decision_logits.reshape(-1, self.config.num_long_classes).type(torch.float32).contiguous()
                speed_decision_loss = decision_criterion(speed_decision_logits, speed_action_targets)

                decision_loss = route_decision_loss + speed_decision_loss

            else:
                actions = actions.reshape(-1, self.config.act_dim).type(torch.float32).contiguous()
                action_targets = torch.argmax(actions, dim=1)
                decision_logits = decision_logits.reshape(-1, self.config.act_dim).type(torch.float32).contiguous()
                decision_loss = decision_criterion(decision_logits, action_targets)
            
            scale = 1.0
            for i in range(trajectory.shape[1]):
                trajectory[:,i] = trajectory[:,i] * scale
                wp_logits[:,i] = wp_logits[:,i] * scale
            
            trajectory = trajectory.reshape(-1, self.config.num_regressions).type(torch.float32).contiguous()
            wp_logits = wp_logits.reshape(-1, self.config.num_regressions).type(torch.float32).contiguous()
            waypoint_loss = trajectory_criterion(wp_logits, trajectory) 
            
            loss = decision_loss + waypoint_loss + prompt_loss.sum()
            
            decision_loss_detached = decision_loss.detach().cpu().numpy()
            decision_loss_sum += decision_loss_detached
            
            waypoint_loss_detached = waypoint_loss.detach().cpu().numpy()
            waypoint_loss_sum += waypoint_loss_detached

            optimizer.zero_grad()

            if not math.isfinite(loss.item()):
                print("Loss is {}, gradscaler training".format(loss.item()))
                scaler.scale(loss.type(torch.float32)).backward()
                scaler.step(optimizer)
                scaler.update()
                lr_scheduler.step()
                torch.cuda.synchronize()
                # sys.exit(1)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                optimizer.step()
                lr_scheduler.step()
                torch.cuda.synchronize()
            
        waypont_loss_mean = waypoint_loss_sum / len(data_loader)
        decision_loss_mean = decision_loss_sum / len(data_loader)
        return decision_loss_mean, waypont_loss_mean
    
    def train_and_evaluate(self, train=True):
        
        # create matrix to save end-of-task accuracies 
        acc_matrix = np.zeros((self.config.num_tasks, self.config.num_tasks))
        route_acc_matrix = np.zeros((self.config.num_tasks, self.config.num_tasks))
        speed_acc_matrix = np.zeros((self.config.num_tasks, self.config.num_tasks))
        now = datetime.datetime.now()

        wandb.login(key="8726d9823ea1bc5190c32369f254d61eab02a17b")

        for task_id in np.arange(self.config.start_task, self.config.num_tasks):
            print(f"Start training for {self.args.epochs[task_id]} epochs over task {task_id}")

            # set task id for model (needed for prompting)
            try:
                self.model.module.task_id = task_id
            except:
                self.model.task_id = task_id

            dont_log_wandb = self.config.dont_log_wandb
            wandb_mode = "online" if not dont_log_wandb else "disabled"

            wandb_name = self.task_name[task_id] + '_' + self.config.prompt_name + '_' + str(self.config.encoder) + '_' + str(self.config.decoder) + '_mobilevlm_mask2former_batch' + str(self.args.batch_size) + '_query16_' + str(self.args.lr_max) + '_' + str(self.args.lr_prompt_max) + '_' + self.task_name[0] + '_' + self.task_name[1] + '_' + self.task_name[2]
            wandb.init(
            project = self.config.project,
            group = self.config.wandb_group,
            name = wandb_name,
            config = {},
            )
            print('wandb initialization:', wandb_name)

            # Freeze the parameters after the first task
            if train:
  
                if task_id > 0:
                    try:
                        if self.model.module.prompt is not None:
                            self.model.module.prompt.process_task_count()
                    except:
                        if self.model.prompt is not None:
                            self.model.prompt.process_task_count()
                    
                    for n, p in self.model.named_parameters():
                        if self.args.vlm_freeze in n:
                            p.requires_grad = False
                        if self.args.encoder_freeze in n:
                            p.requires_grad = False
                        if self.args.decoder_freeze in n:
                            p.requires_grad = False
                        if p.requires_grad == True:
                            print(n)
                else:
                    
                    for n, p in self.model.named_parameters():
                        if self.args.encoder_freeze in n:
                            p.requires_grad = False
                        if self.args.llm_freeze in n:
                            p.requires_grad = False 
                        if p.requires_grad == True:
                            print(n)
            else:
                try:
                    if self.model.module.prompt is not None:
                        self.model.module.prompt.task_count = self.config.num_tasks - 1
                except:
                    if self.model.prompt is not None:
                        self.model.prompt.task_count = self.config.num_tasks - 1

            model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
            params = sum([np.prod(p.size()) for p in model_parameters])
            print ('Total trainable parameters: ', params)
            
            params_dict = [
                    {'params': [p for n, p in self.model.named_parameters() if ('prompt' not in n) and p.requires_grad]},
                    {'params': [p for n, p in self.model.named_parameters() if ('prompt' in n) and p.requires_grad],
                      'lr': self.args.lr_prompt},
                ]
            optimizer = torch.optim.AdamW(params_dict, lr=self.args.lr, weight_decay=1e-5)
            lr_scheduler = OneCycleLR(optimizer=optimizer, max_lr= [self.args.lr_max,  self.args.lr_prompt_max], #0.001, 0.01 #0.00001, 0.0001 for 84 prompts
                                      steps_per_epoch=len(self.data_loader[task_id]['train']),
                                      epochs=self.args.epochs[task_id], pct_start=0.3, cycle_momentum=False)
            
            scaler = GradScaler()
            ##################################################################
            
            if train:
                
                folder_name = self.config.prompt_name + '_' + self.config.encoder\
                                   + '_' + self.config.decoder + '_seed' + str(self.args.seed) + '_mobilevlm_mask2former_batch' + str(self.args.batch_size) + '_' + self.task_name[0] + '_' + self.task_name[1] + '_' + self.task_name[2] + '_' + str(self.args.lr_max) + '_' + str(self.args.lr_prompt_max) + '_' + str(now.year) + str(now.month) + str(now.day) + str(now.hour) + str(now.minute) + str(now.second)
                output_dir = os.path.join(self.args.output_dir, folder_name)
                
                logdir = os.path.join(self.args.logdir, self.task_name[task_id], folder_name)

                # Create logdir
                if not os.path.isdir(logdir):
                    print('Created dir:', logdir)
                    os.makedirs(logdir, exist_ok=True)
                    
                # Create output dir
                if not os.path.isdir(output_dir):
                    print('Created dir:', output_dir)
                    os.makedirs(output_dir, exist_ok=True)
                
                # training 
                for epoch in tqdm(range(self.args.epochs[task_id])): 
                    
                    train_decision_loss, train_waypoint_loss =\
                        self.train_one_epoch(model=self.model,
                                             data_loader=self.data_loader[task_id]['train'],
                                             scaler=scaler, optimizer=optimizer, lr_scheduler=lr_scheduler, 
                                             device=self.device, epoch=epoch, max_norm=self.args.clip_grad, 
                                             set_training_mode=True, task_id=task_id,
                                             class_mask=self.class_mask, args=self.args,)

                    wandb.log({'decision_' + self.config.prompt_name + '_' + self.config.encoder
                                       + '_' + self.config.decoder: train_decision_loss}, step=epoch)
        
                    wandb.log({'waypoint_' + self.config.prompt_name + '_' + self.config.encoder
                                       + '_' + self.config.decoder: train_waypoint_loss}, step=epoch)

                    wandb.log({'foundation_lr_' + self.config.prompt_name + '_' + self.config.encoder
                                       + '_' + self.config.decoder: lr_scheduler.get_last_lr()[0]}, step=epoch)

                    wandb.log({'prompt_lr_' + self.config.prompt_name + '_' + self.config.encoder
                                        + '_' + self.config.decoder: lr_scheduler.get_last_lr()[1]}, step=epoch)

                    if epoch > 0 and epoch % self.args.val_every == 0:
                        task_mean_acc = self.evaluate_till_now(self.model, self.data_loader, self.device,
                                                task_id=task_id, class_mask=self.class_mask,
                                                acc_matrix=acc_matrix, route_acc_matrix=route_acc_matrix,
                                                speed_acc_matrix=speed_acc_matrix, args=self.args, record_dir=output_dir)
                        
                        if task_mean_acc > self.save_threshold:
                            self.save_threshold = task_mean_acc
                            checkpoint_path = os.path.join(output_dir,
                                                            '{}_task{}_epoch{}_acc{:.1f}_checkpoint.pth'.format(self.task_name[task_id], task_id+1, epoch, task_mean_acc))
                            state_dict = {
                                    'model': self.model.state_dict(),
                                    'optimizer': optimizer.state_dict(),
                                    'epoch': epoch,
                                    'args': self.args,
                                }

                            if self.args.sched is not None and self.args.sched != 'constant':
                                state_dict['lr_scheduler'] = self.lr_scheduler.state_dict()
                                
                            torch.save(state_dict, checkpoint_path)
            else:
                epoch = 0
                output_dir = None
                # folder_name = self.config.prompt_name + '_' + self.config.encoder\
                #                     + '_' + self.config.decoder + '_seed' + str(self.args.seed) + '_joint_' + str(now.year) + str(now.month) + str(now.day) + str(now.hour) + str(now.minute) + str(now.second)
                # output_dir = os.path.join(self.args.output_dir, folder_name)
                # os.makedirs(output_dir, exist_ok=True)
            
            print('Final Evaluation')
            self.evaluate_till_now(self.model, self.data_loader, self.device,
                                   task_id=task_id, class_mask=self.class_mask,
                                   acc_matrix=acc_matrix, route_acc_matrix=route_acc_matrix,
                                   speed_acc_matrix=speed_acc_matrix, args=self.args, record_dir=output_dir)
            
            wandb.finish()

            if train:
                checkpoint_path = os.path.join(output_dir,
                                               '{}_task{}_final_checkpoint.pth'.format(self.task_name[task_id], task_id+1))
                state_dict = {
                        'model': self.model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch': epoch,
                        'args': self.args,
                    }

                if self.args.sched is not None and self.args.sched != 'constant':
                    state_dict['lr_scheduler'] = self.lr_scheduler.state_dict()
                    
                torch.save(state_dict, checkpoint_path)

    @torch.no_grad()
    def evaluate_till_now(self, model: torch.nn.Module, data_loader, device,
                          task_id=-1, class_mask=None, acc_matrix=None, 
                          route_acc_matrix=None, speed_acc_matrix=None,
                          args=None, record_dir=None):

        stat_matrix =  np.zeros((8, self.config.num_tasks))

        now = datetime.datetime.now()

        df = pd.DataFrame()

        for i in range(task_id+1):

            print("Start evaluate task:", i)
            
            test_stats, recorded_data = self.evaluate(model=model, data_loader=data_loader[i]['val'], 
                                       device=device, task_id=i, class_mask=class_mask, args=args)
    
            stat_matrix[0, i] = test_stats['DecisionAccuracy']
            stat_matrix[1, i] = test_stats['RouteDecisionAccuracy']
            stat_matrix[2, i] = test_stats['SpeedDecisionAccuracy']
            stat_matrix[3, i] = test_stats['DecisionLoss']
            stat_matrix[4, i] = test_stats['TrajectoryLoss']
            stat_matrix[5, i] = test_stats['traj_ade']
            stat_matrix[6, i] = test_stats['traj_fde']
            stat_matrix[7, i] = test_stats['traj_miss']

            acc_matrix[i, task_id] = test_stats['DecisionAccuracy']
            route_acc_matrix[i, task_id] = test_stats['RouteDecisionAccuracy']
            speed_acc_matrix[i, task_id] = test_stats['SpeedDecisionAccuracy']        
        
            df = df.append(recorded_data)
        
        result_file = record_dir + '/' + self.task_name[task_id] + "_task" + str(task_id+1) + '_' + str(now.year) + str(now.month) + str(now.day) + str(now.hour) + str(now.minute) + str(now.second) + str(self.args.epochs) + "_attended_gpt2_trainall.csv"
        df.to_csv(result_file)
        
        namespace = ['Route Decision']
        
        for i in range(len(namespace)):
            avg_stat = np.divide(np.sum(stat_matrix, axis=1), task_id+1)
        
            diagonal = np.diag(acc_matrix)
        
            result_str = "[Average accuracy till task{}]\t {} Acc: {:.4f} \t {} Loss: {:.4f}".format(task_id+1, namespace[i], avg_stat[0], namespace[i], avg_stat[1])
            if task_id > 0:
                
                forgetting = np.mean((np.max(acc_matrix, axis=1) -
                                    acc_matrix[:, task_id])[:task_id])
                backward = np.mean((acc_matrix[:, task_id] - diagonal)[:task_id])
        
                result_str += "\tForgetting: {:.4f}\tBackward: {:.4f}".format(forgetting, backward)

            print(result_str)
            print("Accuracy Matrix:", acc_matrix)
            print("Route Accuracy Matrix:", route_acc_matrix)
            print("Speed Accuracy Matrix:", speed_acc_matrix)
            print("Best Accuracy", diagonal)
            print("Decision Loss:", stat_matrix[3, :])
            print("Trajectory Loss:", stat_matrix[4, :])
            print('ADE:', stat_matrix[5, :])
            print('FDE:', stat_matrix[6, :])
            print('Miss Rate:', stat_matrix[7, :])
            
            data = {
                "task sequence": self.task_name,
                "acc_matrix": acc_matrix.tolist(),
                "route_acc_matrix": route_acc_matrix.tolist(),
                "speed_acc_matrix": speed_acc_matrix.tolist(),
                "Best Accuracy": diagonal.tolist(),
                "Decision Loss": stat_matrix[3, :].tolist(),
                "Trajectory Loss": stat_matrix[4, :].tolist(),
                'ADE': stat_matrix[5, :].tolist(),
                "FDE": stat_matrix[6, :].tolist(),
                "Miss Rate": stat_matrix[7, :].tolist(),
                "Start Task ID": self.config.start_task,
                }

            if record_dir:
                result_file = record_dir + '/' + self.task_name[task_id] + "_task" + str(task_id+1) + '_' + str(now.year) + str(now.month) + str(now.day) + str(now.hour) + str(now.minute) + str(now.second) + str(self.args.epochs) + "_attended_gpt2_trainall.json"
                f = open(result_file, "w")
                json.dump(data, f, indent=4)
                f.close()
                
        mean_acc = np.mean(acc_matrix, axis=0)
        task_mean_acc = mean_acc[task_id]
        return task_mean_acc
    
    @torch.no_grad()
    def evaluate(self, model: torch.nn.Module, data_loader, device, task_id=-1,
                 class_mask=None, args=None,):
        
        trajectory_criterion = torch.nn.MSELoss().to(self.device)
        decision_criterion = torch.nn.CrossEntropyLoss().to(self.device)

        metric_logger = utils.MetricLogger(delimiter="  ")
        header = 'Test: [Task {}]'.format(task_id + 1)

        # switch to evaluation mode
        model.eval()

        with torch.no_grad():
            decision_loss_sum = 0.0
            waypoint_loss_sum = 0.0
            route_decision_acc = 0.0
            speed_decision_acc = 0.0
            decision_acc = 0.0
            
            route_deci_acc = []
            speed_deci_acc = []
            deci_acc = []
            traj_ade = []
            traj_fde = []
            traj_miss_rate = []
            
            for it, data in enumerate(data_loader):
                
                states = data['states'].to(device)
                rtgs = data['rtgs'].to(device)
                timesteps = data['timesteps'].to(device)
                detections = data['detections'].to(device)
                route_actions = data['route_decisions'].to(device)
                speed_actions = data['speed_decisions'].to(device)
                decisions = None
                target_waypoints = data['target_waypoints'].to(device)
                reference_path = data['reference_path'].to(device)
                ego_path = data['ego_path'].to(device)
                masks = data['masks'].to(device)
                
                if self.config.label == 'local_planning':
                    trajectory = ego_path
                else:
                    trajectory = reference_path
    
                if self.config.decision == 'dual':
                    actions = torch.cat((route_actions, speed_actions), dim=-1)
                else:
                    actions = route_actions

                # combine the batch dimension and seq_len dimension
                states = states.reshape(-1, 3, self.config.img_resolution[1],
                                        self.config.img_resolution[0]).type(torch.float32).contiguous()
                decision_logits, wp_logits, _ = model.forward(states, actions, rtgs, timesteps, detections,
                                                              decisions, target_waypoints=target_waypoints, task_id=task_id)
                
                if self.config.decision == 'dual':
                    route_actions = route_actions.reshape(-1, self.config.num_lat_classes).type(torch.float32).contiguous()
                    route_action_targets = torch.argmax(route_actions, dim=1)
                    route_decision_logits = decision_logits[:,:,:self.config.num_lat_classes]
                    route_decision_logits = route_decision_logits.reshape(-1, self.config.num_lat_classes).type(torch.float32).contiguous()
                    route_decision_loss = decision_criterion(route_decision_logits, route_action_targets)
    
                    speed_actions = speed_actions.reshape(-1, self.config.num_long_classes).type(torch.float32).contiguous()
                    speed_action_targets = torch.argmax(speed_actions, dim=1)
                    speed_decision_logits = decision_logits[:,:,self.config.num_lat_classes:]
                    speed_decision_logits = speed_decision_logits.reshape(-1, self.config.num_long_classes).type(torch.float32).contiguous()
                    speed_decision_loss = decision_criterion(speed_decision_logits, speed_action_targets)
    
                    decision_loss = route_decision_loss + speed_decision_loss

                    [route_acc] = accuracy(route_decision_logits, route_action_targets, topk=(1,))
                    route_decision_acc += route_acc
                    route_deci_acc.append(route_acc.detach().cpu().numpy())
                    
                    [speed_acc] = accuracy(speed_decision_logits, speed_action_targets, topk=(1,))
                    speed_decision_acc += speed_acc
                    speed_deci_acc.append(speed_acc.detach().cpu().numpy())
                    
                    deci_acc.append((route_acc.detach().cpu().numpy() + speed_acc.detach().cpu().numpy())/2)
    
                else:
                    actions = actions.reshape(-1, self.config.act_dim).type(torch.float32).contiguous()
                    action_targets = torch.argmax(actions, dim=1)
                    decision_logits = decision_logits.reshape(-1, self.config.act_dim).type(torch.float32).contiguous()
                    decision_loss = decision_criterion(decision_logits, action_targets)
                
                    [route_acc] = accuracy(decision_logits, action_targets, topk=(1,))
                    route_decision_acc += route_acc
                    route_deci_acc.append(route_acc.detach().cpu().numpy())
                    
                    speed_decision_acc += route_acc
                    speed_deci_acc.append(route_acc.detach().cpu().numpy())
                    
                    deci_acc.append((route_acc.detach().cpu().numpy() + route_acc.detach().cpu().numpy())/2)
                
                ade, fde = utils.displacement_error(wp_logits, trajectory)
                traj_ade.append(ade)
                traj_fde.append(fde)

                trajectory = trajectory.reshape(-1, self.config.num_regressions).type(torch.float32).contiguous()
                wp_logits = wp_logits.reshape(-1, self.config.num_regressions).type(torch.float32).contiguous()
                waypoint_loss = trajectory_criterion(wp_logits, trajectory) 
                
                miss_rate = utils.miss_rate(wp_logits, trajectory)
                traj_miss_rate.append(miss_rate)
                
                decision_loss_detached = decision_loss.detach().cpu().numpy()
                decision_loss_sum += decision_loss_detached
                
                waypoint_loss_detached = waypoint_loss.detach().cpu().numpy()
                waypoint_loss_sum += waypoint_loss_detached
            
            waypont_loss_mean = waypoint_loss_sum / len(data_loader)
            decision_loss_mean = decision_loss_sum / len(data_loader)
            route_decision_acc_mean = route_decision_acc / len(data_loader)
            speed_decision_acc_mean = speed_decision_acc / len(data_loader)
            decision_acc_mean = (route_decision_acc_mean + speed_decision_acc_mean)/2

            metric_logger.meters['TrajectoryLoss'].update(waypont_loss_mean.item())
            metric_logger.meters['DecisionLoss'].update(decision_loss_mean.item())
            metric_logger.meters['RouteDecisionAccuracy'].update(route_decision_acc_mean.item(), n=it)
            metric_logger.meters['SpeedDecisionAccuracy'].update(speed_decision_acc_mean.item(), n=it)
            metric_logger.meters['DecisionAccuracy'].update(decision_acc_mean.item(), n=it)
            metric_logger.meters['traj_ade'].update(np.mean(traj_ade))
            metric_logger.meters['traj_fde'].update(np.mean(traj_fde))
            metric_logger.meters['traj_miss'].update(np.mean(traj_miss_rate))

        # gather the stats from all processes
        metric_logger.synchronize_between_processes()

        print('\n Task {} and its decision loss {:.3f} and waypoint loss {:.3f}:'.format(self.task_name[task_id], decision_loss_mean, waypont_loss_mean))

        df = pd.DataFrame(data={'route_decision_accuracy': route_deci_acc, 'speed_decision_accuracy': speed_deci_acc,
                                'decision_accuracy': deci_acc,'trajectory_ade:': traj_ade, 'trajectory_fde': traj_fde,
                                'trajectory_miss_rate': traj_miss_rate})
                                
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, df
