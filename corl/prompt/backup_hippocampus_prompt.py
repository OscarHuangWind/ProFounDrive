#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 15:18:07 2024

@author: oscar
"""

import torch
import torch.nn as nn

import copy
import numpy as np

class HiPrompt(nn.Module):
    def __init__(self, emb_d, n_tasks, prompt_param, key_dim=768):
        super().__init__()
        self.task_count = 0
        self.emb_d = emb_d
        self.key_d = key_dim
        self.n_tasks = n_tasks
        self._init_smart(emb_d, prompt_param)
        self.attend = nn.Softmax(dim = -1)
        self.prompt_criterion = torch.nn.CrossEntropyLoss()
        # e prompt init
        for e in self.e_layers:
            # for model saving/loading simplicity, we init the full paramaters here
            # however, please note that we reinit the new components at each task
            # in the "spirit of continual learning", as we don't know how many tasks
            # we will encounter at the start of the task sequence
            #
            # in the original paper, we used ortho init at the start - this modification is more 
            # fair in the spirit of continual learning and has little affect on performance
            e_l = self.e_p_length
            p = tensor_prompt(self.e_pool_size, e_l, emb_d)
            k = tensor_prompt(self.e_pool_size, self.key_d)
            # a = tensor_prompt(self.e_pool_size, self.key_d)
            p = self.gram_schmidt(p)
            k = self.gram_schmidt(k)
            # a = self.gram_schmidt(a)
            setattr(self, f'e_p_{e}',p)
            setattr(self, f'e_k_{e}',k)
            # setattr(self, f'e_a_{e}',a)

    def prompt_loss(self, prediction, target):
        loss = self.prompt_criterion(prediction, target)
        return loss

    def _init_smart(self, emb_d, prompt_param):

        # prompt basic param
        self.e_pool_size = int(prompt_param[0])
        self.e_p_length = int(prompt_param[1])
        self.e_layers = [0,1,2,3,4] # -1 for visualprompt_dt

        # strenth of ortho penalty
        self.ortho_mu = prompt_param[2]
        
    def process_task_count(self):
        self.task_count += 1

        # in the spirit of continual learning, we will reinit the new components
        # for the new task with Gram Schmidt
        #
        # in the original paper, we used ortho init at the start - this modification is more 
        # fair in the spirit of continual learning and has little affect on performance
        # 
        # code for this function is modified from:
        # https://github.com/legendongary/pytorch-gram-schmidt/blob/master/gram_schmidt.py
        for e in self.e_layers:
            K = getattr(self,f'e_k_{e}')
            # A = getattr(self,f'e_a_{e}')
            P = getattr(self,f'e_p_{e}')
            k = self.gram_schmidt(K)
            # a = self.gram_schmidt(A)
            p = self.gram_schmidt(P)
            setattr(self, f'e_p_{e}',p)
            setattr(self, f'e_k_{e}',k)
            # setattr(self, f'e_a_{e}',a)

    # code for this function is modified from:
    # https://github.com/legendongary/pytorch-gram-schmidt/blob/master/gram_schmidt.py
    def gram_schmidt(self, vv):

        def projection(u, v):
            denominator = (u * u).sum()

            if denominator < 1e-8:
                return None
            else:
                return (v * u).sum() / denominator * u

        # check if the tensor is 3D and flatten the last two dimensions if necessary
        is_3d = len(vv.shape) == 3
        if is_3d:
            shape_2d = copy.deepcopy(vv.shape)
            vv = vv.view(vv.shape[0],-1)

        # swap rows and columns
        vv = vv.T

        # process matrix size
        nk = vv.size(1)
        uu = torch.zeros_like(vv, device=vv.device)

        # get starting point
        pt = int(self.e_pool_size / (self.n_tasks))
        s = int(self.task_count * pt)
        f = int((self.task_count + 1) * pt)
        if s > 0:
            uu[:, 0:s] = vv[:, 0:s].clone()
        for k in range(s, f):
            redo = True
            while redo:
                redo = False
                vk = torch.randn_like(vv[:,k]).to(vv.device)
                uk = 0
                for j in range(0, k):
                    if not redo:
                        uj = uu[:, j].clone()
                        proj = projection(uj, vk)
                        if proj is None:
                            redo = True
                            print('restarting!!!')
                        else:
                            uk = uk + proj
                if not redo: uu[:, k] = vk - uk
        for k in range(s, f):
            uk = uu[:, k].clone()
            uu[:, k] = uk / (uk.norm())

        # undo swapping of rows and columns
        uu = uu.T 

        # return from 2D
        if is_3d:
            uu = uu.view(shape_2d)
        
        return torch.nn.Parameter(uu) 

    def forward(self, x_querry, l, x_block=None, train=False, task_id=None):

        # e prompts
        e_valid = False
        if l in self.e_layers:
            e_valid = True
            B, C = x_querry.shape

            K = getattr(self,f'e_k_{l}')
            # A = getattr(self,f'e_a_{l}')
            p = getattr(self,f'e_p_{l}')
            pt = int(self.e_pool_size / (self.n_tasks))
            s = int(self.task_count * pt)
            f = int((self.task_count + 1) * pt)
            
            # querry 
            q = nn.functional.normalize(x_querry, dim=1).detach()
            local_p_list = []
            global_attention_list = []
            
            if train:
                
                # attention
                if self.task_count > 0:               
                    for start in range(0, s, pt):
                        end = start + pt
                        key = K[start:end].detach().clone()
                        value = p[start:end].detach().clone()
                        n_key = nn.functional.normalize(key, dim=1)
                        cos_sim = torch.einsum('bj,kj->bk', q, n_key)
                        # local_attention = nn.functional.normalize(cos_sim, dim=1)
                        local_attention = self.attend(cos_sim)
                        p_ = torch.einsum('bk,kld->bld', local_attention, value)
                        local_p_list.append(p_)
                        
                        n_p_ = nn.functional.normalize(p_, dim=1)
                        cos_sim = torch.einsum('bj,kj->bk', q, n_p_)
                        global_attention = nn.functional.normalize(cos_sim, dim=1)
                        #global_attention = self.attend(cos_sim)
                        global_attention_list.append(global_attention)
                        
                    key_curr = K[s:f]
                    value_curr = p[s:f]
                    n_key_curr = nn.functional.normalize(key_curr, dim=1)
                    cos_sim_curr = torch.einsum('bj,kj->bk', q, n_key_curr)
                    # attention_curr = nn.functional.normalize(cos_sim_curr, dim=1)
                    attention_curr = self.attend(cos_sim_curr)
                    p_curr = torch.einsum('bk,kld->bld', attention_curr, value_curr)
                    local_p_list.append(p_curr)
                    
                    with torch.no_grad():
                        n_p_curr = nn.functional.normalize(p_curr, dim=1)
                        cos_sim = torch.einsum('bj,kj->bk', q, n_p_curr)
                        global_attention = nn.functional.normalize(cos_sim, dim=1)
                        #global_attention = self.attend(cos_sim)
                        global_attention_list.append(global_attention)
                    
                    local_p = np.array(local_p_list)
                    global_attention = np.array(global_attention_list)
                    
                    max_idx = np.argmax(global_attention)
                    global_p = local_p[:, max_idx]
                    
                    target_domain = np.zeros_like(global_attention)
                    target_domain[:,self.task_count] = 1
                    loss = self.prompt_loss(global_attention, target_domain)
                    
                else:
                    key_curr = K[s:f]
                    value_curr = p[s:f]
                    n_key_curr = nn.functional.normalize(key_curr, dim=1)
                    cos_sim_curr = torch.einsum('bj,kj->bk', q, n_key_curr)
                    # attention_curr = nn.functional.normalize(cos_sim_curr, dim=1)
                    attention_curr = self.attend(cos_sim_curr)
                    p_curr = torch.einsum('bk,kld->bld', attention_curr, value_curr)
                    global_p = p_curr
                    
                    loss = 0
            else:
                loss = 0
                with torch.no_grad():
                    for start in range(0, f, pt):
                        end = start + pt
                        key = K[start:end]
                        value = p[start:end]
                        n_key = nn.functional.normalize(key, dim=1)
                        cos_sim = torch.einsum('bj,kj->bk', q, n_key)
                        # local_attention = nn.functional.normalize(cos_sim, dim=1)
                        attention_curr = self.attend(cos_sim)
                        p_ = torch.einsum('bk,kld->bld', local_attention, value)
                        local_p_list.append(p_)
                        
                        n_p_ = nn.functional.normalize(p_, dim=1)
                        cos_sim = torch.einsum('bj,kj->bk', q, n_p_)
                        global_attention = nn.functional.normalize(cos_sim, dim=1)
                        #global_attention = self.attend(cos_sim)
                        global_attention_list.append(global_attention)
                        
                    local_p = np.array(local_p_list)
                    global_attention = np.array(global_attention_list)
                    
                    max_idx = np.max(global_attention)
                    global_p = local_p[:, max_idx]

            # select prompts
            i = int(self.e_p_length/2)
            Ek = global_p[:,:i,:]
            Ev = global_p[:,i:,:]

            # ortho penalty
            # if train and self.ortho_mu > 0:
            #     loss = ortho_penalty(K) * self.ortho_mu
            #     loss += ortho_penalty(A) * self.ortho_mu
            #     loss += ortho_penalty(p.view(p.shape[0], -1)) * self.ortho_mu
            # else:
            #     loss = 0
        else:
            loss = 0

        # combine prompts for prefix tuning
        if e_valid:
            p_return = [Ek, Ev]
        else:
            p_return = None

        # return
        return p_return, loss, x_block

def tensor_prompt(a, b, c=None, ortho=False):
    if c is None:
        p = torch.nn.Parameter(torch.DoubleTensor(a,b), requires_grad=True)
    else:
        p = torch.nn.Parameter(torch.DoubleTensor(a,b,c), requires_grad=True)
    if ortho:
        nn.init.orthogonal_(p)
    else:
        # nn.init.uniform_(p)
        torch.nn.init.xavier_uniform_(p)
    return p 