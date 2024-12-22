#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 22:46:42 2024

@author: oscar
"""

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

from .trajectory_gpt2 import GPT2Model, GPT2LMHeadModel
from .gpt_tokenizer import VLMAdapter

class PromptGPT(nn.Module):

    def __init__(
            self,
            config,
            state_dim,
            act_dim,
            embed_size,
            hidden_size,
            seq_len=None,
            embed_len=3,
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
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.img_resolution = 224
        self.img_state_dim = self.img_resolution**2 * 3
        
        #### load pretrained Decision Transformer model ####
        # dt_model = DecisionTransformerModel.from_pretrained("edbeeching/decision-transformer-gym-hopper-medium")
        
        #### load pretrained Q-former ####
        self.vlm_adapter = VLMAdapter( num_token_queries=16, llm_hidden_dim=self.hidden_size)

        #### load pretrained GPT model ######
        gpt_config = transformers.GPT2Config(vocab_size=1, n_embd=hidden_size, **kwargs)
        llm_model = GPT2Model(gpt_config)
        # llm_model = GPT2LMHeadModel(gpt_config)
        
        if self.config.mode == 'onboard':
            if self.config.decoder == 'GPT2-xl':
                self.llm_model = llm_model.from_pretrained('gpt2-xl', torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
            else:
                self.llm_model = llm_model.from_pretrained('gpt2', torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)

            self.embed_vlm_state = torch.nn.Linear(self.hidden_size, self.hidden_size, dtype=torch.bfloat16)
            self.embed_image_state = torch.nn.Linear(self.config.encoder_embed_dim, embed_size, dtype=torch.bfloat16)
            self.embed_goal_state = torch.nn.Linear(self.config.num_regressions, self.hidden_size, dtype=torch.bfloat16)
            self.embed_return = torch.nn.Linear(1, self.hidden_size, dtype=torch.bfloat16)
            self.embed_action = torch.nn.Linear(self.act_dim, self.hidden_size, dtype=torch.bfloat16)
            self.embed_ln = nn.LayerNorm(self.hidden_size, dtype=torch.bfloat16)
            self.embed_timestep = nn.Embedding(max_emb_size, self.hidden_size, dtype=torch.bfloat16)

            # self.project_action = torch.nn.Linear(embed_size, self.hidden_size, dtype=torch.bfloat16)
            # self.project_rtg = torch.nn.Linear(embed_size, self.hidden_size, dtype=torch.bfloat16)
            
            self.predict_class = nn.Sequential(
                *(
                    [nn.Linear(self.hidden_size, self.act_dim, dtype=torch.bfloat16)] + 
                    ([nn.Softmax(dim=2)] if softmax else []))
            )

        else:
            if self.config.decoder == 'GPT2-xl':
                self.llm_model = llm_model.from_pretrained('gpt2-xl', low_cpu_mem_usage=True)
            else:
                self.llm_model = llm_model.from_pretrained('gpt2', low_cpu_mem_usage=True)

            self.embed_vlm_state = torch.nn.Linear(self.hidden_size, self.hidden_size, dtype=torch.float32)
            self.embed_image_state = torch.nn.Linear(self.config.encoder_embed_dim, embed_size, dtype=torch.float32)
            self.embed_goal_state = torch.nn.Linear(self.config.num_regressions, self.hidden_size, dtype=torch.float32)
            self.embed_return = torch.nn.Linear(1, self.hidden_size, dtype=torch.float32)
            self.embed_action = torch.nn.Linear(self.act_dim, self.hidden_size, dtype=torch.float32)
            self.embed_ln = nn.LayerNorm(self.hidden_size, dtype=torch.float32)
            self.embed_timestep = nn.Embedding(max_emb_size, self.hidden_size, dtype=torch.float32)
            
            # self.project_action = torch.nn.Linear(embed_size, self.hidden_size, dtype=torch.float32)
            # self.project_rtg = torch.nn.Linear(embed_size, self.hidden_size, dtype=torch.float32)
    
            self.predict_class = nn.Sequential(
                *(
                    [nn.Linear(self.hidden_size, self.act_dim, dtype=torch.float32)] + 
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

        image_state_embeddings = self.embed_image_state(states)
        image_token_embeddings = image_state_embeddings.permute(0, 2, 1, 3).reshape(batch_size, seq_length, -1, self.embed_size)
        input_dicts = {'LanguageInstruction': condition_states, 'ImageTokenEmbedding': image_token_embeddings}

        # get language embedded stacked outputs from Q-former
        get_embeddings_func = self.llm_model.get_input_embeddings()
        out, multimodal_attention_mask = self.vlm_adapter(input_dicts, get_embeddings_func, train=train)
        
        # time embeddings are treated similar to positional embeddings
        # condition_state_embeddings = condition_state_embeddings + time_embeddings
        # image_state_embeddings = image_state_embeddings + time_embeddings

        # condition_state_embeddings = self.embed_condition_state(condition_states)
        rtg_embeddings = self.embed_return(returns_to_go).unsqueeze(2)
        vlm_embeddings = self.embed_vlm_state(out)
        goal_embeddings = self.embed_goal_state(target_waypoints).unsqueeze(2)
        state_embeddings = torch.cat((goal_embeddings, vlm_embeddings), dim=-2)
        action_embeddings = self.embed_action(actions).unsqueeze(2)
        time_embeddings = self.embed_timestep(timesteps.reshape(-1, self.seq_len)).unsqueeze(2)

        action_embeddings = action_embeddings + time_embeddings
        rtg_embeddings = rtg_embeddings + time_embeddings
        state_embeddings = state_embeddings + time_embeddings

        llm_embeddings = torch.cat((rtg_embeddings, state_embeddings, action_embeddings), dim=-2)
        llm_inputs = self.embed_ln(llm_embeddings).view(batch_size, -1, self.hidden_size)

        # KDP attention_mask B Prompt_len * seq_len
        # prompt_attention_mask = torch.ones((batch_size, seq_length)).to(states.device)
        # prompt_len = int(self.prompt_len / 2) # half for key and value
        # prompt_attention_mask = prompt_attention_mask.repeat(1, prompt_len)
        # rtg_attention_mask = attention_mask.repeat(1, rtg_embeddings.shape[-2])
        # action_attention_mask = attention_mask.repeat(1, action_embeddings.shape[-2])

        rtg_attention_mask = attention_mask.unsqueeze(-1).repeat(1, 1, rtg_embeddings.shape[-2])
        goal_attention_mask = attention_mask.unsqueeze(-1).repeat(1, 1, goal_embeddings.shape[-2])
        state_attention_mask = torch.cat((goal_attention_mask, multimodal_attention_mask), dim=-1)
        action_attention_mask = attention_mask.unsqueeze(-1).repeat(1, 1, action_embeddings.shape[-2])
        
        if prompt is not None:
            prompt_len = int(self.prompt_len / 2) # half for key and value
            prompt_attention_mask = torch.ones((batch_size, seq_length, prompt_len)).to(states.device)

            llm_attention_mask = torch.cat((prompt_attention_mask,
                                            rtg_attention_mask,
                                            state_attention_mask,
                                            action_attention_mask), dim=-1).view(batch_size, -1)
        else:
            llm_attention_mask = torch.cat((rtg_attention_mask,
                                            state_attention_mask,
                                            action_attention_mask), dim=-1).view(batch_size, -1)            

        transformer_outputs, prompt_loss = self.llm_model(
                                                inputs_embeds=llm_inputs,
                                                attention_mask=llm_attention_mask,
                                                prompt=prompt,
                                                q=q,
                                                train=train,
                                                task_id=task_id,
                                            )

        x = transformer_outputs['last_hidden_state']

        x = x.reshape(batch_size, seq_length, -1, self.hidden_size).permute(0, 2, 1, 3)

        # if prompt is None:
        #     # reshape x so that the second dimension corresponds to the original
        #     # returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for s_t
        #     x = x.reshape(batch_size, seq_length, -1, self.hidden_size).permute(0, 2, 1, 3)
        # else:
        #     x = x.reshape(batch_size, seq_length, -1, self.hidden_size).permute(0, 2, 1, 3)
        
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
            attention_mask = attention_mask.to(dtype=torch.float32, device=timesteps.device).reshape(1, -1)

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
                              dtype=torch.float32,
                              device=actions.device), actions], dim=1)
           
            returns_to_go = torch.cat(
                [torch.zeros((returns_to_go.shape[0], 
                              self.seq_len-returns_to_go.shape[1], 
                              returns_to_go.shape[-1]),
                              dtype=torch.float32,
                              device=returns_to_go.device), returns_to_go], dim=1)
            
            timesteps = torch.cat(
                [torch.zeros((timesteps.shape[0], 
                              self.seq_len-timesteps.shape[1], 
                              timesteps.shape[-1]), 
                             device=timesteps.device), timesteps], dim=1).to(torch.int32)
            
            detections = torch.cat(
                [torch.zeros((detections.shape[0], 
                              self.seq_len-detections.shape[1], 
                              detections.shape[-1]), dtype=torch.float32,
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
                                  target_waypoints.shape[-1]), dtype=torch.float32,
                                  device=target_waypoints.device), target_waypoints], dim=1)
            
        else:
            attention_mask = None
    
        # Note: prompt within kwargs
        action_preds, _, x = self.forward(
            img_states, actions, returns_to_go, timesteps, detections, decisions=decisions,
            target_waypoints=target_waypoints, attention_mask=attention_mask,
            prompt=prompt, q=q, train=train, task_id=task_id)
    
        return _, action_preds, _, x #waypoint_pred
