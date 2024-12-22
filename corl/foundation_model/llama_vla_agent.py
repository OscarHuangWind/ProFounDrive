#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 21:28:15 2024

@author: oscar
"""

import cv2
import timm
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel

import contextlib

from timm import create_model
from transformers import AutoTokenizer, LlamaConfig
from .MobileVLM.mask2former import SegEncoder

from prompt.prompt_baseline import CodaPrompt, DualPrompt, L2P
from prompt.drive_prompt import DrivePrompt

from .MobileVLM.mobilevlm import PromptVLM
# from transformers import LlamaTokenizer
# from .MobileVLM.mobilevlm import load_pretrained_model
# from .MobileVLM.mobilellama import MobileLlamaForCausalLM

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""
    
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)
    

class MobileVLA_Agent(nn.Module):
    def __init__(self, config):
        super(MobileVLA_Agent, self).__init__()

        self.config = config

        # prompt
        self.prompt_name = config.prompt_name
        self.task_id = None

        # architecture
        self.encoder = config.encoder
        self.decoder = config.decoder
        
        # vision foundation model parameter
        self.img_size = config.img_resolution[0]
        self.patch_size = config.patch_size
        self.encoder_embed_dim = config.encoder_embed_dim
        self.depth = config.depth
        self.num_heads = config.num_heads
        self.num_classes = config.act_dim
        self.num_lat_classes = config.num_lat_classes
        self.num_regressions = config.num_regressions
        self.depth = config.depth
        self.ckpt_layer = config.ckpt_layer
        self.drop_path_rate = config.drop_path_rate
        
        # large language model parameter
        self.seq_len = config.seq_len
        self.pred_len = config.pred_len
        self.embed_len = config.embed_len
        self.prompt_len = config.e_prompt_len
        self.decoder_embed_dim = config.decoder_embed_dim
        self.hidden_dim = config.hidden_dim
        self.state_dim = config.state_dim
        self.max_emb_size = config.max_emb_size
        self.n_layer = config.n_layer
        self.n_head = config.n_head
        self.dropout = config.dropout
        
        # segformer
        if config.image_type == 'segmentation':
            self.seg_encoder = SegEncoder(self.config.device)
        else:
            self.seg_encoder = None
        
        self.prompt_vlm = PromptVLM(
            config=config,
            state_dim=self.state_dim, 
            act_dim=self.num_classes,
            seq_len=self.seq_len,
            prompt_len=self.prompt_len,
            max_emb_size=self.max_emb_size,
            hidden_size=self.hidden_dim,
            ).to(config.device)

        # trajectory decoder
        self.trajectory_head = nn.GRUCell(input_size=self.config.num_lat_classes + self.num_regressions + 2, # 2 represents x,y coordinate, 6 represents road option
                                  hidden_size=self.hidden_dim)
        self.last = nn.Linear(self.hidden_dim, config.num_regressions)
        self.softmax = nn.Softmax(dim=2)
        self.last_act = nn.Tanh()

        if self.config.mode == 'onboard':
            self.trajectory_head.to(dtype=torch.bfloat16)
            self.last.to(dtype=torch.bfloat16)

        # create learnable prompt module
        if self.prompt_name == 'l2p':
            prompt_param = [config.num_tasks, [config.pool_size, config.e_prompt_len, config.prompt_depth, config.top_k]]
            self.prompt = L2P(self.encoder_embed_dim, prompt_param[0], prompt_param[1], key_dim=self.hidden_dim, device=config.device)
        elif self.prompt_name == 'dual':
            prompt_param = [config.num_tasks, [config.pool_size, config.e_prompt_len, config.g_prompt_len, config.top_k]]
            self.prompt = DualPrompt(self.encoder_embed_dim, prompt_param[0], prompt_param[1], key_dim=self.hidden_dim, device=config.device)
        elif self.prompt_name == 'coda':
            prompt_param = [config.num_tasks, [config.pool_size, config.e_prompt_len, config.g_prompt_len, config.ortho]]
            self.prompt = CodaPrompt(self.encoder_embed_dim, prompt_param[0], prompt_param[1], key_dim=self.hidden_dim, device=config.device)
        elif self.prompt_name == 'drive':
            print('Drive Prompt')
            prompt_param = [config.num_tasks, [config.pool_size, config.e_prompt_len, config.top_k, config.attended_p]]
            self.prompt = DrivePrompt(self.encoder_embed_dim, prompt_param[0], prompt_param[1], key_dim=self.hidden_dim, mode= config.mode, device=config.device)
        else:
            print('No prompt: ViT + DT')
            self.prompt = None

    def maybe_autocast(self, dtype=torch.float16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        # enable_autocast = self.device != torch.device("cpu")
        enable_autocast = True

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    # pen: get penultimate vit_encoderures    
    def forward(self, states, actions=None, rtgs=None, timesteps=None,
                detections=None, decisions=None, target_waypoints=None,
                attention_mask=None, pen=False, train=False, task_id=None):

        with self.maybe_autocast():
            #### the states are already preprocessed (RGB or Segmentation) ####
            with torch.no_grad():
                image_tensor = self.prompt_vlm.image_processor.preprocess(states, return_tensors='pt')['pixel_values']

            out, prompt_loss, logit = self.prompt_vlm(image_tensor, actions, rtgs, timesteps,
                                            detections, decisions=decisions,
                                            target_waypoints=target_waypoints,
                                            attention_mask=attention_mask,
                                            prompt=self.prompt, task_id=task_id,
                                            train=train,
                                            ) 
            
            if self.config.decision == 'dual':
            
                route_softmax_out = self.softmax(out[:,:,:self.config.num_lat_classes])
                route_index = torch.argmax(route_softmax_out, dim=2)
                last_route_decision = F.one_hot(route_index, num_classes=self.config.num_lat_classes)[:,-1,:]

                speed_softmax_out = self.softmax(out[:,:,self.config.num_lat_classes:])
                speed_index = torch.argmax(speed_softmax_out, dim=2)
                last_speed_decision = F.one_hot(speed_index, num_classes=self.config.num_long_classes)[:,-1,:]               
                
                last_decision = last_route_decision.detach()
                last_logit = logit[:,-1,:]

            else:
                softmax_out = self.softmax(out)
                out_index = torch.argmax(softmax_out, dim=2)
                
                last_decision = F.one_hot(out_index, num_classes=6)[:,-1,:].detach()
                last_logit = logit[:,-1,:]
            
            trajectory = self.forward_gru(last_logit, last_decision, target_waypoints[:,-1,:])                    
            
            return out, trajectory, prompt_loss        

    def forward_gru(self, z, decision_out, target_waypoint):
        output_wp = list()
        batch_size = z.shape[0]
        # initial input variable to GRU
        x = torch.zeros(size=(batch_size, 2), dtype=z.dtype).to(z.device)
        decision_out = decision_out.reshape(-1, self.config.num_lat_classes)
        
        # autoregressive generation of output waypoints
        for _ in range(self.pred_len):
            x_in = torch.cat([x, target_waypoint, decision_out], dim=-1)
            z_in = z     
            z = self.trajectory_head(x_in, z_in)
            dx = self.last(z)

            x = dx + x
            
            output_wp.append(x)
            
        pred_wp = torch.stack(output_wp, dim=1)
        return pred_wp

    def get_action(self, states, actions=None, rtgs=None, timesteps=None,
                   detections=None, decisions=None, target_waypoints=None,
                   attention_mask=None, pen=False, train=False, task_id=None):

        with self.maybe_autocast():
            #### the states are already preprocessed (RGB or Segmentation) ####
            with torch.no_grad():
                image_tensor = self.prompt_vlm.image_processor.preprocess(states, return_tensors='pt')['pixel_values']

                _, out, _, logit = self.prompt_vlm.get_action(image_tensor, actions, rtgs, timesteps,
                                                                     detections, decisions=decisions,
                                                                     target_waypoints=target_waypoints,
                                                                     prompt=self.prompt, train=train,
                                                                     task_id=task_id,
                                                                    ) 
                
                if self.config.decision == 'dual':
                
                    route_softmax_out = self.softmax(out[:,:,:self.config.num_lat_classes])
                    route_index = torch.argmax(route_softmax_out, dim=2)
                    last_route_decision = F.one_hot(route_index, num_classes=self.config.num_lat_classes)[:,-1,:]
    
                    speed_softmax_out = self.softmax(out[:,:,self.config.num_lat_classes:])
                    speed_index = torch.argmax(speed_softmax_out, dim=2)
                    last_speed_decision = F.one_hot(speed_index, num_classes=self.config.num_long_classes)[:,-1,:]               
                    
                    last_decision = torch.cat((last_route_decision, last_speed_decision), dim=-1).detach()
                    last_logit = logit[:,-1,:]
                    out_index = [route_index, speed_index]

                else:
                    softmax_out = self.softmax(out)
                    out_index = torch.argmax(softmax_out, dim=2)
                    
                    last_decision = F.one_hot(out_index, num_classes=6)[:,-1,:].detach()
                    last_logit = logit[:,-1,:]
                
                trajectory = self.forward_gru(last_logit, last_route_decision.detach(), target_waypoints[:,-1,:])
                print(trajectory)
                return out_index, last_decision, trajectory, route_softmax_out[:,-1,:].cpu().numpy().squeeze(), speed_softmax_out[:,-1,:].cpu().numpy().squeeze()                

def init_llama_vla(config):
    return MobileVLA_Agent(config)
