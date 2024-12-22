# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 15:51:12 2024

@author: Admin
"""

import torch
import torch.nn as nn

import copy
import numpy as np

class HiPrompt(nn.Module):
    def __init__(self, emb_d, n_tasks, prompt_param, key_dim=768, device='cuda:0'):
        super().__init__()
        self.task_count = 0
        self.emb_d = emb_d
        self.value_d = key_dim
        self.n_tasks = n_tasks
        self._init_smart(emb_d, prompt_param)
        self.attend = nn.Softmax(dim = -1)
        self.prompt_criterion = torch.nn.CrossEntropyLoss()
        self.global_p = None
                
        # self.mean = nn.Parameter(torch.zeros(1, device=device))
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
            p = tensor_prompt(self.e_pool_size, e_l, self.emb_d, device=device)
            local_k = tensor_prompt(self.e_pool_size, self.emb_d, device=device)
            global_k = tensor_prompt(self.n_tasks, self.emb_d, device=device)
            # a = tensor_prompt(self.e_pool_size, self.key_d)
            p = self.gram_schmidt(p)
            local_k = self.gram_schmidt(local_k)
            global_k = self.gram_schmidt_global(global_k)
            # a = self.gram_schmidt(a)
            setattr(self, f'e_p_{e}', p)
            setattr(self, f'e_lk_{e}', local_k)
            setattr(self, f'e_gk_{e}', global_k)
            # setattr(self, f'e_a_{e}',a)

    def prompt_loss(self, prediction, target):
        loss = self.prompt_criterion(prediction, target)
        return loss

    def ortho_penalty(self, t):
        return ((t @t.T - torch.eye(t.shape[0]).cuda())**2).mean()

    def _init_smart(self, emb_d, prompt_param):
        self.name = 'corl'
        
        # prompt basic param
        self.e_pool_size = int(prompt_param[0])
        self.e_p_length = int(prompt_param[1])
        self.e_layers = [0,1,2,3,4] # -1 for visualprompt_dt

        self.top_k = prompt_param[-2]
        self.attended_p = prompt_param[-1]
        
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
            local_K = getattr(self,f'e_lk_{e}')
            global_K = getattr(self,f'e_gk_{e}')

            local_k = self.gram_schmidt(local_K)
            global_k = self.gram_schmidt_global(global_K)
            
            P = getattr(self,f'e_p_{e}')
            p = self.gram_schmidt(P)
            
            setattr(self, f'e_p_{e}',p)
            setattr(self, f'e_lk_{e}',local_k)
            setattr(self, f'e_gk_{e}',global_k)

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

    def gram_schmidt_global(self, vv):

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
        s = 0 #int(self.task_count * pt)
        f = self.n_tasks #int((self.task_count + 1) * pt)
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

            local_k = getattr(self,f'e_lk_{l}')
            global_k = getattr(self,f'e_gk_{l}')
            p = getattr(self,f'e_p_{l}')
            pt = int(self.e_pool_size / (self.n_tasks))
            s = int(self.task_count * pt)
            f = int((self.task_count + 1) * pt)
            
            # querry 
            q = nn.functional.normalize(x_querry, dim=1).detach()
            local_p_list = []
            
            if train:
                
                # global attention
                start = int(self.task_count)
                end = int(self.task_count+1)
                # global_key = torch.cat((global_k[:start].detach().clone(),global_k[start:end]), dim=0)
                global_key = global_k[0:end]

                # global_key = global_K
                n_global_key = nn.functional.normalize(global_key, dim=1)
                global_cos_sim = torch.einsum('bj,kj->bk', q, n_global_key)
                global_attention = self.attend(global_cos_sim)
                
                top_k = torch.topk(global_attention, self.top_k, dim=1)
                k_idx = top_k.indices
                loss = (1.0 - global_cos_sim[:,k_idx]).sum() + self.ortho_penalty(global_key)
                # loss = self.ortho_penalty(global_key)
                
                # loss = (1.0 - global_cos_sim[:,self.task_count]).sum()
                
                # global_attention = self.attend(cos_sim)
                
                # with torch.no_grad():
                #     target_domain = torch.zeros_like(global_attention).to(device=global_attention.device)
                #     target_domain[:,self.task_count] = 1
                #     target_domain = torch.argmax(target_domain, dim=1)

                # loss = self.prompt_loss(global_attention, target_domain) * 5 #+ self.ortho_penalty(global_key)
                
                # attention
                
                if self.attended_p and self.task_count > 0:
                    for start in range(0, s, pt):
                        end = start + pt
                        local_key = local_k[start:end].detach().clone()
                        value = p[start:end].detach().clone()
                        n_key = nn.functional.normalize(local_key, dim=1)
                        cos_sim = torch.einsum('bj,kj->bk', q, n_key)
                        # local_attention = nn.functional.normalize(cos_sim, dim=1)
                        local_attention = self.attend(cos_sim)
                        p_ = torch.einsum('bk,kld->bld', local_attention, value)
                        local_p_list.append(p_)
                        
                    key_curr = local_k[s:f]
                    value_curr = p[s:f]
                    n_key_curr = nn.functional.normalize(key_curr, dim=1)
                    cos_sim_curr = torch.einsum('bj,kj->bk', q, n_key_curr)
                    # attention_curr = nn.functional.normalize(cos_sim_curr, dim=1)
                    attention_curr = self.attend(cos_sim_curr)
                    p_curr = torch.einsum('bk,kld->bld', attention_curr, value_curr)
                    local_p_list.append(p_curr)
                    
                    local_p = torch.stack(local_p_list, dim=1) # bpld
                    global_p = torch.einsum("bpld, bp->bld", (local_p, global_attention))

                else:
                    key_curr = local_k[s:f]
                    value_curr = p[s:f]
                    n_key_curr = nn.functional.normalize(key_curr, dim=1)
                    cos_sim_curr = torch.einsum('bj,kj->bk', q, n_key_curr)
                    # attention_curr = nn.functional.normalize(cos_sim_curr, dim=1)
                    attention_curr = self.attend(cos_sim_curr)
                    p_curr = torch.einsum('bk,kld->bld', attention_curr, value_curr)
                    global_p = p_curr

                # if self.task_count > 0:
                    
                #     # for start in range(0, s, pt):
                #     #     end = start + pt
                #     #     local_key = local_K[start:end].detach().clone()
                #     #     value = p[start:end].detach().clone()
                #     #     n_key = nn.functional.normalize(local_key, dim=1)
                #     #     cos_sim = torch.einsum('bj,kj->bk', q, n_key)
                #     #     # local_attention = nn.functional.normalize(cos_sim, dim=1)
                #     #     local_attention = self.attend(cos_sim)
                #     #     p_ = torch.einsum('bk,kld->bld', local_attention, value)
                #     #     local_p_list.append(p_)
                        
                #     key_curr = local_K[s:f]
                #     value_curr = p[s:f]
                #     n_key_curr = nn.functional.normalize(key_curr, dim=1)
                #     cos_sim_curr = torch.einsum('bj,kj->bk', q, n_key_curr)
                #     # attention_curr = nn.functional.normalize(cos_sim_curr, dim=1)
                #     attention_curr = self.attend(cos_sim_curr)
                #     p_curr = torch.einsum('bk,kld->bld', attention_curr, value_curr)
                #     global_p = p_curr
                #     # local_p_list.append(p_curr)
                #     # local_p = np.array(local_p_list)
                    
                # else:
                #     key_curr = local_K[s:f]
                #     value_curr = p[s:f]
                #     n_key_curr = nn.functional.normalize(key_curr, dim=1)
                #     cos_sim_curr = torch.einsum('bj,kj->bk', q, n_key_curr)
                #     # attention_curr = nn.functional.normalize(cos_sim_curr, dim=1)
                #     attention_curr = self.attend(cos_sim_curr)
                #     p_curr = torch.einsum('bk,kld->bld', attention_curr, value_curr)
                #     global_p = p_curr
                    
            else:
                # inference 
                loss = 0
                with torch.no_grad():
                    
                    # global attention
                    start = int(self.task_count)
                    end = int(self.task_count+1)
                    global_key = global_k[0:end] #torch.cat((global_K[:start].detach().clone(),global_K[start:end]), dim=0)

                    # global_key = global_K
                    n_global_key = nn.functional.normalize(global_key, dim=1)
                    global_cos_sim = torch.einsum('bj,kj->bk', q, n_global_key)
                    global_attention = self.attend(global_cos_sim)
                    
                    # top_k = torch.topk(global_cos_sim, self.top_k, dim=1)
                    # selected_domain = top_k.indices
                    
                    for start in range(0, f, pt):
                        end = start + pt
                        local_key = local_k[start:end]
                        value = p[start:end]
                        n_key = nn.functional.normalize(local_key, dim=1)
                        cos_sim = torch.einsum('bj,kj->bk', q, n_key)
                        # local_attention = nn.functional.normalize(cos_sim, dim=1)
                        local_attention = self.attend(cos_sim)
                        p_ = torch.einsum('bk,kld->bld', local_attention, value)
                        local_p_list.append(p_)
                    
                    local_p = torch.stack(local_p_list, dim=1) # bpld
                    
                    if self.attended_p:
                        global_p = torch.einsum("bpld, bp->bld", (local_p, global_attention))
                        self.global_p = global_p
                    else:
                        selected_domain = torch.argmax(global_attention, dim=1)
                        global_p = local_p[range(local_p.shape[0]), selected_domain,:,:]
                        # global_p = local_p[range(local_p.shape[0]), selected_domain.squeeze(1),:,:]
                        self.global_p = global_p

                    # if self.task_count > 0:

                    #     for start in range(0, f, pt):
                    #         end = start + pt
                    #         local_key = local_K[start:end]
                    #         value = p[start:end]
                    #         n_key = nn.functional.normalize(local_key, dim=1)
                    #         cos_sim = torch.einsum('bj,kj->bk', q, n_key)
                    #         # local_attention = nn.functional.normalize(cos_sim, dim=1)
                    #         local_attention = self.attend(cos_sim)
                    #         p_ = torch.einsum('bk,kld->bld', local_attention, value)
                    #         local_p_list.append(p_)
                        
                    #     local_p = torch.stack(local_p_list, dim=1) # bpld

                        # if self.attend_p:
                        #     global_p = torch.einsum("bpld, bp->bld", (local_p, global_attention))
                        # else:
                        #     selected_domain = torch.argmax(global_attention, dim=1)
                        #     global_p = local_p[selected_domain]
                    # else:
                        
                        # key_curr = local_K[s:f]
                        # value_curr = p[s:f]
                        # n_key_curr = nn.functional.normalize(key_curr, dim=1)
                        # cos_sim_curr = torch.einsum('bj,kj->bk', q, n_key_curr)
                        # # attention_curr = nn.functional.normalize(cos_sim_curr, dim=1)
                        # attention_curr = self.attend(cos_sim_curr)
                        # p_curr = torch.einsum('bk,kld->bld', attention_curr, value_curr)
                        # global_p = p_curr

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

def tensor_prompt(a, b, c=None, ortho=False, device='cudo:0'):
    if c is None:
        p = torch.nn.Parameter(torch.DoubleTensor(a,b).to(device), requires_grad=True)
    else:
        p = torch.nn.Parameter(torch.DoubleTensor(a,b,c).to(device), requires_grad=True)
    if ortho:
        nn.init.orthogonal_(p)
    else:
        # nn.init.uniform_(p)
        torch.nn.init.xavier_uniform_(p)
    return p 