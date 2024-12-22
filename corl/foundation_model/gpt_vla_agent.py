#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 16:20:26 2024

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

from .GptVLM.vision_encoder.mask2former import SegEncoder
from .GptVLM.vision_encoder.vit import VisionTransformer
from .GptVLM.decoder.language_decoder import PromptGPT

from prompt.prompt_baseline import CodaPrompt, DualPrompt, L2P
from prompt.driving_skill_prompt import DSPrompt

class GPTVLA_Agent(nn.Module):

    def __init__(self, config):
        super(GPTVLA_Agent, self).__init__()

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
        
        # pre-trained encoder
        if self.encoder == 'CLIP-ViT-Large':
            self.vit_encoder = timm.create_model('vit_huge_patch14_clip_224.laion2b_ft_in12k_in1k', pretrained=True, num_classes=0)#.to(dtype=torch.bfloat16)
            if config.mode == 'onboard':
                self.vit_encoder.to(dtype=torch.bfloat16)
        elif self.encoder == 'ViT-Tiny':
            self.vit_encoder = VisionTransformer(img_size=self.img_size,
                                          patch_size=self.patch_size,
                                          embed_dim=self.encoder_embed_dim,
                                          depth=self.depth,
                                          num_heads=self.num_heads,
                                          ckpt_layer=self.ckpt_layer,
                                          drop_path_rate=self.drop_path_rate
                                        )
            
            from timm.models import vit_tiny_patch16_224
            load_dict = vit_tiny_patch16_224(pretrained=True).state_dict()
            del load_dict['head.weight']; del load_dict['head.bias']
            
            self.vit_encoder.load_state_dict(load_dict)

            if config.mode == 'onboard':
                self.vit_encoder.to(dtype=torch.bfloat16)

        # create prompting module
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
            self.prompt = DSPrompt(self.encoder_embed_dim, prompt_param[0], prompt_param[1], key_dim=self.hidden_dim, mode= config.mode, device=config.device)
        else:
            print('No prompt: ViT + DT')
            self.prompt = None
    
        if self.decoder:
            # Decision Transformer decoder and FC classifier
            self.prompt_gpt = PromptGPT(
                config=config,
                state_dim=self.state_dim, 
                act_dim=self.num_classes,
                seq_len=self.seq_len,
                embed_len=self.embed_len,
                prompt_len=self.prompt_len,
                max_emb_size=self.max_emb_size,
                embed_size=self.decoder_embed_dim,
                hidden_size=self.hidden_dim,
                n_layer=self.n_layer,
                n_head=self.n_head,
                n_inner=4*self.hidden_dim,
                activation_function='relu',
                n_positions=1024,
                resid_pdrop=self.dropout,
                attn_pdrop=self.dropout
                ).to(config.device)

        self.trajectory_head = nn.GRUCell(input_size=self.config.num_lat_classes + self.num_regressions + 2, # 2 represents x,y coordinate, 6 represents road option
                                  hidden_size=self.hidden_dim)
        self.last = nn.Linear(self.hidden_dim, config.num_regressions)
        self.softmax = nn.Softmax(dim=2)
        self.last_act = nn.Tanh()
        
        if self.config.mode == 'onboard':
            self.trajectory_head.to(dtype=torch.bfloat16)
            self.last.to(dtype=torch.bfloat16)

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
                if self.encoder == 'CLIP-ViT-Large':
                    out = self.vit_encoder.forward_features(states)
                elif self.encoder == 'ViT-Tiny':
                    out, _ = self.vit_encoder(states)

            if self.prompt is not None: # drive, l2p, dual, coda,

                if self.decoder: # prompt + encoder + decoder + gru; language drive
                    q = out[:,0,:] # batch*seq_len, encoder_embed_dim
                    features = out[:,1:,:] # batch*seq_len, token_len, encoder_embed_dim               

                    features = features.reshape(actions.shape[0], self.seq_len, -1, self.encoder_embed_dim) # batch, seq_len, token_len, encoder_embed_dim
                
                    out, prompt_loss, logit = self.prompt_gpt(features, actions, rtgs, timesteps,
                                                    detections, decisions=decisions,
                                                    target_waypoints=target_waypoints,
                                                    attention_mask=attention_mask,
                                                    prompt=self.prompt, q=q, 
                                                    task_id=task_id, train=train,
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
                                
                else: # prompt + encoder + gru; not for language drive
        
                    q = out[:,0,:]
                    out, prompt_loss = self.vit_encoder(states, prompt=self.prompt,
                                                        q=q, train=train, task_id=self.task_id)
                    
                    out = out[:,0,:]
            
                    if decisions is None:
                        condition_states = torch.cat((target_waypoints, detections), dim=-1)
                    elif target_waypoints is None:
                        condition_states = torch.cat((decisions, detections), dim=-1)
                    else:
                        condition_states = detections
                    
                    condition_state_embeddings = self.condition_embed(condition_states).reshape(-1, self.encoder_embed_dim)
                    logit = torch.cat((out, condition_state_embeddings), dim=-1)
                    out = self.last_act(self.last(logit))
                    
                    softmax_out = self.softmax(out)
                    out_index = torch.argmax(softmax_out, dim=2)
                    with torch.no_grad():
                        last_decision = F.one_hot(out_index, num_classes=6)[:,-1,:]
                        
                    last_logit = logit[:,-1,:]
                    
                    trajectory = self.forward_gru(last_logit, last_decision, target_waypoints[:,-1,:])
        
                return out, trajectory, prompt_loss        

            else: # no prompt

                if self.decoder: # encoder + decoder + gru; language drive
                    q = out[:,0,:] # batch*seq_len, encoder_embed_dim
                    features = out[:,1:,:] # batch*seq_len, token_len, encoder_embed_dim               
                    
                    features = features.reshape(actions.shape[0], self.seq_len, -1, self.encoder_embed_dim) # batch, seq_len, token_len, encoder_embed_dim
                    
                    out, prompt_loss, logit = self.prompt_gpt(features, actions, rtgs, timesteps,
                                                    detections, decisions=decisions,
                                                    target_waypoints=target_waypoints,
                                                    attention_mask=attention_mask,
                                                    prompt=self.prompt, q=q, 
                                                    task_id=task_id, train=train,
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
                    
                else: # encoder + gru; not for language drive
            
                    out = out[:,0,:]
            
                    if decisions is None:
                        condition_states = torch.cat((target_waypoints, detections), dim=-1)
                    elif target_waypoints is None:
                        condition_states = torch.cat((decisions, detections), dim=-1)
                    else:
                        condition_states = detections
        
                    condition_state_embeddings = self.condition_embed(condition_states).reshape(-1, self.encoder_embed_dim)
                    logit = torch.cat((out, condition_state_embeddings), dim=-1)
                    out = self.last_act(self.last(logit))
                    
                    softmax_out = self.softmax(out)
                    out_index = torch.argmax(softmax_out, dim=2)
                    with torch.no_grad():
                        last_decision = F.one_hot(out_index, num_classes=6)[:,-1,:]
                    
                    last_logit = logit[:,-1,:]
                    
                    trajectory = self.forward_gru(last_logit, last_decision, target_waypoints[:,-1,:])

                return out, trajectory, _

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
                   attention_mask=None, pen=False, train=False):
        
        with self.maybe_autocast():

            if self.prompt is not None: # corl, l2p, dual, coda, l2p_dt, dual_dt, coda_dt

                if self.decoder:
                    with torch.no_grad():

                        if self.encoder == 'CLIP-ViT-Large':
                            out = self.vit_encoder.forward_features(states)
                        elif self.encoder == 'ViT-Tiny':
                            out, _ = self.vit_encoder(states)

                        q = out[:,0,:] # batch*seq_len, encoder_embed_dim
                        features = out[:,1:,:] # batch*seq_len, token_len, encoder_embed_dim               
                        features = features.reshape(actions.shape[0], self.seq_len, -1, self.encoder_embed_dim) # batch, seq_len, token_len, encoder_embed_dim
                    
                    _, out, _, logit = self.prompt_gpt.get_action(features, actions, rtgs, timesteps,
                                                                detections, decisions=decisions, 
                                                                target_waypoints=target_waypoints,
                                                                prompt=self.prompt,
                                                                q=q, train=train, task_id=self.task_id)

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
                    
                    with torch.autocast(device_type='cuda', dtype=torch.float32):
                        trajectory = self.forward_gru(last_logit, last_route_decision.detach(), target_waypoints[:,-1,:])
                    
                    return out_index, last_decision, trajectory, route_softmax_out[:,-1,:].cpu().numpy().squeeze(), speed_softmax_out[:,-1,:].cpu().numpy().squeeze()
                
                else: # ViT + prompt
                    with torch.no_grad():
                        features, _ = self.vit_encoder(states)
                        q = features[:,0,:]
                    out, prompt_loss = self.vit_encoder(states, prompt=self.prompt,
                                                        q=q, train=train, task_id=self.task_id)
                    
                    return out_index, last_decision, None

            else:
                if self.decoder:
                    with torch.no_grad():

                        if self.encoder == 'CLIP-ViT-Large':
                            out = self.vit_encoder.forward_features(states)
                        elif self.encoder == 'ViT-Tiny':
                            out, _ = self.vit_encoder(states)

                        q = out[:,0,:] # batch*seq_len, encoder_embed_dim
                        features = out[:,1:,:] # batch*seq_len, token_len, encoder_embed_dim               
                        features = features.reshape(actions.shape[0], self.seq_len, -1, self.encoder_embed_dim) # batch, seq_len, token_len, encoder_embed_dim
                    
                    _, out, _, logit = self.prompt_gpt.get_action(features, actions, rtgs, timesteps,
                                                                detections, decisions=decisions, 
                                                                target_waypoints=target_waypoints,
                                                                prompt=self.prompt,
                                                                q=q, train=train, task_id=self.task_id)

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
                    
                    with torch.autocast(device_type='cuda', dtype=torch.float32):
                        trajectory = self.forward_gru(last_logit, last_route_decision.detach(), target_waypoints[:,-1,:])
                    
                    return out_index, last_decision, trajectory, route_softmax_out[:,-1,:].cpu().numpy().squeeze(), speed_softmax_out[:,-1,:].cpu().numpy().squeeze()
                
                else: # ViT
                    out, _ = self.vit_encoder(states)
                    out = out[:,0,:]
                
                    softmax_out = self.softmax(out)
                    out_index = torch.argmax(softmax_out, dim=-1)
                    last_decision = F.one_hot(out_index, num_classes=6)[:,-1,:]
            
                    return out_index, last_decision, None

def init_gpt_vla(config):
    return GPTVLA_Agent(config)
