#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 16:18:17 2024

@author: oscar
"""

# Code backbone: Decision Transformer https://github.com/kzl/decision-transformer/
# Decision Transformer License: https://github.com/kzl/decision-transformer/blob/master/LICENSE.md

import numpy as np
import torch
import torch.nn as nn

import transformers
# from transformers import DecisionTransformerModel

from .trajectory_gpt2 import GPT2Model

class PromptGPT(nn.Module):

    def __init__(
            self,
            config,
            state_dim,
            act_dim,
            hidden_size,
            seq_len=None,
            embed_len=4,
            prompt_len=10,
            max_emb_size=8192, #4096,
            action_tanh=False,
            softmax=False,
            **kwargs
    ):
        super().__init__()
        self.config = config
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.seq_len = seq_len
        self.embed_len = embed_len
        self.prompt_len = prompt_len
        self.hidden_size = hidden_size
        self.img_resolution = 224
        self.img_state_dim = self.img_resolution**2 * 3
        
        #### load pretrained Decision Transformer model ####
        # dt_model = DecisionTransformerModel.from_pretrained("edbeeching/decision-transformer-gym-hopper-medium")
        
        #### load pretrained GPT model ######
        gpt_config = transformers.GPT2Config(vocab_size=1, n_embd=hidden_size, **kwargs)
        foundation_model = GPT2Model(gpt_config)
        
        if self.config.mode == 'onboard':
            self.foundation_model = foundation_model.from_pretrained('gpt2-xl', torch_dtype=torch.bfloat16)
            self.embed_timestep = nn.Embedding(max_emb_size, hidden_size, dtype=torch.bfloat16)
            self.embed_image_state = torch.nn.Linear(self.config.embed_dim, hidden_size, dtype=torch.bfloat16)
            self.embed_condition_state = torch.nn.Linear(self.state_dim, hidden_size, dtype=torch.bfloat16)
            self.embed_return = torch.nn.Linear(1, hidden_size, dtype=torch.bfloat16)
            self.embed_action = torch.nn.Linear(self.act_dim, hidden_size, dtype=torch.bfloat16)

            self.embed_ln = nn.LayerNorm(hidden_size, dtype=torch.bfloat16)
    
            self.predict_class = nn.Sequential(
                *(
                    [nn.Linear(hidden_size, self.act_dim, dtype=torch.bfloat16)] + 
                    ([nn.Softmax(dim=2)] if softmax else []))
            )

        else:
            self.foundation_model = foundation_model.from_pretrained('gpt2')
            self.embed_timestep = nn.Embedding(max_emb_size, hidden_size)
            self.embed_image_state = torch.nn.Linear(self.config.embed_dim, hidden_size)
            self.embed_condition_state = torch.nn.Linear(self.state_dim, hidden_size)
            self.embed_return = torch.nn.Linear(1, hidden_size)
            self.embed_action = torch.nn.Linear(self.act_dim, hidden_size)
            
            self.embed_ln = nn.LayerNorm(hidden_size)

            self.predict_class = nn.Sequential(
                *(
                    [nn.Linear(hidden_size, self.act_dim)] + 
                    ([nn.Softmax(dim=2)] if softmax else []))
            )

    def forward(self, states, actions, returns_to_go, timesteps,
                detections, decisions, target_waypoints, attention_mask=None,
                prompt=None, q=None, train=None, task_id=None):
        batch_size, seq_length = states.shape[0], states.shape[1]
        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length)).to(states.device)

        # embed each modality with a different head
        if decisions is None:
            condition_states = torch.cat((target_waypoints, detections), dim=-1)
        elif target_waypoints is None   : 
            condition_states = torch.cat((decisions, detections), dim=-1)
        
        condition_state_embeddings = self.embed_condition_state(condition_states)
        image_state_embeddings = self.embed_image_state(states)
        action_embeddings = self.embed_action(actions)
        rtg_embeddings = self.embed_return(returns_to_go)
        time_embeddings = self.embed_timestep(timesteps.reshape(-1, self.seq_len))

        # time embeddings are treated similar to positional embeddings
        condition_state_embeddings = condition_state_embeddings + time_embeddings
        image_state_embeddings = image_state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        rtg_embeddings = rtg_embeddings + time_embeddings

        stacked_inputs = torch.stack(
            (rtg_embeddings, image_state_embeddings,
              condition_state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, self.embed_len*seq_length, self.hidden_size)
        stacked_inputs = self.embed_ln(stacked_inputs)

        # to make the attention mask fit the stacked inputs, have to stack it as well
        prompt_attention_mask = torch.ones((batch_size, seq_length)).to(states.device)
        prompt_len = int(self.prompt_len / 2)
        stacked_prompt_attention_mask = prompt_attention_mask.repeat(1, prompt_len)
        
        stacked_state_attention_mask = torch.stack(
            (attention_mask, attention_mask, attention_mask, attention_mask), dim=1
        ).permute(0, 2, 1).reshape(batch_size, self.embed_len*seq_length)
        stacked_attention_mask = torch.cat((stacked_prompt_attention_mask, stacked_state_attention_mask), dim=-1)
        
        transformer_outputs, prompt_loss = self.foundation_model(
                                                inputs_embeds=stacked_inputs,
                                                attention_mask=stacked_attention_mask,
                                                prompt=prompt,
                                                q=q,
                                                train=train,
                                                task_id=task_id,
                                            )
        x = transformer_outputs['last_hidden_state']

        if prompt is None:
            # reshape x so that the second dimension corresponds to the original
            # returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for s_t
            x = x.reshape(batch_size, seq_length, 4, self.hidden_size).permute(0, 2, 1, 3)
        else:
            x = x.reshape(batch_size, -1, 4, self.hidden_size).permute(0, 2, 1, 3)
        
        # layer normalization
        # x = self.embed_ln(x)
        
        # note here all the prompt are pre-append to x, but when return only return the last [:, -seq_length:, :] corresponding to batch data
        # get predictions
        # return_preds = self.predict_return(x[:,1])[:, -seq_length:, :]  # predict next return given state and action
        action_preds = self.predict_class(x[:,-2])[:, -seq_length:, :]  # predict next action given state
                
        if torch.any(torch.isnan(action_preds)):
            print('lat wtf?')

        return action_preds, prompt_loss, x[:,-2] #, waypoint_pred

    def get_action(self, img_states, actions, returns_to_go, timesteps,
                   detections, decisions=None, target_waypoints=None, prompt=None,
                   q=None, train=None, task_id=None):
    
        if self.seq_len is not None:
            img_states = img_states[:,-self.seq_len:]
            actions = actions[:,-self.seq_len:]
            returns_to_go = returns_to_go[:,-self.seq_len:]
            timesteps = timesteps[:,-self.seq_len:]
            detections = detections[:,-self.seq_len:]
            
            if decisions is not None:
                decisions = decisions[:,-self.seq_len:]
    
            if target_waypoints is not None:
                target_waypoints = target_waypoints[:,-self.seq_len:]
    
            # pad all tokens to sequence length
            attention_mask = torch.cat([torch.zeros(self.seq_len-actions.shape[1]), torch.ones(actions.shape[1])])
            attention_mask = attention_mask.to(dtype=torch.bfloat16, device=timesteps.device).reshape(1, -1)

            # actions = torch.cat(
            #     [torch.zeros((actions.shape[0], 
            #                   self.seq_len - actions.shape[1], 
            #                   self.act_dim),
            #                  device=actions.device), actions],
            #     dim=1).to(dtype=torch.float64)
           
            actions = torch.cat(
                [torch.zeros((actions.shape[0], 
                              self.seq_len - actions.shape[1], 
                              self.act_dim),
                              dtype=torch.bfloat16,
                              device=actions.device), actions], dim=1)
           
            returns_to_go = torch.cat(
                [torch.zeros((returns_to_go.shape[0], 
                              self.seq_len-returns_to_go.shape[1], 
                              returns_to_go.shape[-1]),
                              dtype=torch.bfloat16,
                              device=returns_to_go.device), returns_to_go], dim=1)
            
            timesteps = torch.cat(
                [torch.zeros((timesteps.shape[0], 
                              self.seq_len-timesteps.shape[1], 
                              timesteps.shape[-1]), 
                             device=timesteps.device), timesteps], dim=1).to(torch.int32)
            
            detections = torch.cat(
                [torch.zeros((detections.shape[0], 
                              self.seq_len-detections.shape[1], 
                              detections.shape[-1]), dtype=torch.bfloat16,
                              device=detections.device), detections], dim=1)
            
            if decisions is not None:
                decisions = torch.cat(
                    [torch.zeros((decisions.shape[0], 
                                  self.seq_len-decisions.shape[1], 
                                  decisions.shape[-1]), dtype=torch.int16,
                                  device=decisions.device), decisions], dim=1)
                
            if target_waypoints is not None:
                target_waypoints = torch.cat(
                    [torch.zeros((target_waypoints.shape[0], 
                                  self.seq_len-target_waypoints.shape[1], 
                                  target_waypoints.shape[-1]), dtype=torch.bfloat16,
                                  device=target_waypoints.device), target_waypoints], dim=1)
            
        else:
            attention_mask = None
    
        # Note: prompt within kwargs
        action_preds, _, x = self.forward(
            img_states, actions, returns_to_go, timesteps, detections, decisions=decisions,
            target_waypoints=target_waypoints, attention_mask=attention_mask,
            prompt=prompt, q=q, train=train, task_id=task_id)
    
        return _, action_preds, _, x #waypoint_pred
