# ------------------------------------------
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
# ------------------------------------------
# Modification:
# Added code for Simple Continual Learning datasets
# -- Jaeho Lee, dlwogh9344@khu.ac.kr
# ------------------------------------------

import random
import numpy as np
import torch
from torch.utils.data import random_split
from torch.utils.data.dataset import Subset

import utils
from dataset_preprocess.data import CARLA_Data

def target_transform(x, nb_classes):
    return x + nb_classes

# We need to seed the workers individually otherwise random processes in the dataloader return the same values across workers!
def seed_worker(worker_id):
    # Torch initial seed is properly set across the different workers, we need to pass it to numpy and random.
    worker_seed = (torch.initial_seed()) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def build_continual_dataloader(args, config):
    dataloader = list()

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True # Wen want the highest performance
    generator = torch.Generator().manual_seed(seed)

    train_dataset_root = config.train_data
    val_dataset_root = config.val_data



    # for i in range(3): # for joint training
    for i in range(config.num_tasks):
        train_town_dataset_root = []
        val_town_dataset_root = []
        if args.dataset_mode.startswith('all'): # joint Learning

            if 'Split' in args.dataset_mode:
                carla_dataset = CARLA_Data(root=train_dataset_root, config=config, shared_dict=None)

                # train 90% val 10%                    
                train_size = int(0.9*len(carla_dataset))
                val_size = len(carla_dataset) - train_size
                carla_train_dataset, carla_val_dataset = random_split(carla_dataset, [train_size, val_size], generator=generator)
            
            else:
                carla_train_dataset = CARLA_Data(root=train_dataset_root, config=config, shared_dict=None)
                carla_val_dataset = CARLA_Data(root=val_dataset_root, config=config, shared_dict=None)
        else:
            town_list = args.task_seq 
        
            args.nb_classes = 0
                
            for j in train_dataset_root:
                if town_list[i] in j:
                    train_town_dataset_root.append(j)
            
            if args.dataset_mode.startswith('Split-'): # Split-Sequential: validation on same datasets
                    carla_dataset = CARLA_Data(root=train_town_dataset_root, config=config, shared_dict=None)
                        
                    # train 90% val 10%                    
                    train_size = int(0.9*len(carla_dataset))
                    val_size = len(carla_dataset) - train_size
                    carla_train_dataset, carla_val_dataset = random_split(carla_dataset, [train_size, val_size], generator=generator)
                    
            else: # Sequential: validation on new datasets
                
                for j in val_dataset_root:
                    if town_list[i] in j:
                        val_town_dataset_root.append(j)
                
                carla_train_dataset = CARLA_Data(root=train_town_dataset_root, config=config, shared_dict=None)
                # print(len(carla_train_dataset))
                for i in range(len(carla_train_dataset)):
                    carla_train_dataset[i]
                carla_val_dataset = CARLA_Data(root=val_town_dataset_root, config=config, shared_dict=None)
            
        sampler_train = torch.utils.data.RandomSampler(carla_train_dataset)
        sampler_val = torch.utils.data.SequentialSampler(carla_val_dataset)
        
        data_loader_train = torch.utils.data.DataLoader(
            carla_train_dataset, sampler=sampler_train,
            batch_size=args.batch_size,
            worker_init_fn=seed_worker,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            # collate_fn=collate_fn,
        )

        data_loader_val = torch.utils.data.DataLoader(
            carla_val_dataset, sampler=sampler_val,
            worker_init_fn=seed_worker,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
        )

        dataloader.append({'train': data_loader_train, 'val': data_loader_val})
        # dataloader.append({'train': data_loader_train})
    
    print('Data Preprocess Done!')
    return dataloader