import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import time
import json
import tqdm
import random
import datetime
import argparse
import numpy as np

import torch
import torch.multiprocessing as mp
from torch.distributed.elastic.multiprocessing.errors import record

from engine import Engine
from corl_config import GlobalConfig
from dataset_preprocess.datasets import build_continual_dataloader
from foundation_model.gpt_vla_agent import init_gpt_vla
from foundation_model.llama_vla_agent import init_llama_vla

# Records error and tracebacks in case of failure
@record
def main():
    torch.cuda.empty_cache()

    parser = argparse.ArgumentParser()
    
    # Train parameters
    parser.add_argument('--train', default=True, type=str, help='Batch size for one GPU. When training with multiple GPUs the effective batch size will be batch_size*num_gpus')
    parser.add_argument('--batch_size', default=16, type=int, help='Batch size for one GPU. When training with multiple GPUs the effective batch size will be batch_size*num_gpus')
    parser.add_argument('--epochs', default=[70, 50, 50], help='Number of train epochs.') # [70, 50, 50]
    parser.add_argument('--val_every', type=int, default=5, help='At which epoch frequency to validate.')

    # Optimizer parameters
    parser.add_argument('--opt', default='adam', type=str, metavar='OPTIMIZER', help='Optimizer (default: "adam"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON', help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=(0.9, 0.999), type=float, nargs='+', metavar='BETA', help='Optimizer Betas (default: (0.9, 0.999), use opt default)')
    parser.add_argument('--clip-grad', type=float, default=1.0, metavar='NORM',  help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.0, help='weight decay (default: 0.0)')
    parser.add_argument('--reinit_optimizer', type=bool, default=True, help='reinit optimizer (default: True)')    
    
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='constant', type=str, metavar='SCHEDULER', help='LR scheduler (default: "constant"')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate.') #1e-4 #1e-5
    parser.add_argument('--lr-prompt', type=float, default=1e-4, help='Learning rate for prompt.') #5e-3 #1e-5
    parser.add_argument('--lr-max', type=float, default=0.001, help='max learning rate for foundaiton models')
    parser.add_argument('--lr-prompt-max', type=float, default=0.005, help='max leaning rate for learnable prompt') #0.005 for seg

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=None, metavar='PCT', help='Color jitter factor (default: 0.3)')
    parser.add_argument('--aa', type=str, default=None, metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')
    
    # Data parameters
    parser.add_argument('--data-path',  default=r'/home/users/ntu/shanhelo/scratch/datasets/carlacorl', help='Root directory of your training data')
    parser.add_argument('--save-path', type=str, default='/home/users/ntu/shanhelo/scratch/wenhui_projects/ProFounDrive/corl/output/', help='ckpt to load.')
    parser.add_argument('--output_dir', default='/home/users/ntu/shanhelo/scratch/wenhui_projects/ProFounDrive/corl/output/', help='path where to save, empty for no saving')    
    parser.add_argument('--dataset_mode', default='Split-sequential', type=str, help='dataset load mode, 1.all 2. all-Split 3.Split-sequential, 4.sequential')
    parser.add_argument('--logdir', type=str, default='log', help='Directory to log data to.')
    parser.add_argument('--load-pretrain', type=bool, default=False, help='ckpt to load.')
    parser.add_argument('--start_epoch', type=int, default=0, help='Epoch to start with. Useful when continuing trainings via load_file.')
    parser.add_argument('--setting', type=str, default='min-3-town', help='What training setting to use. Options: '
                                                                   'all: Train on all towns no validation data. '
                                                                   '3-town: Train on urban(Town3), highway(Town4), rural(Town7) no validation data'
                                                                   '02_05_withheld: Do not train on Town 02 and Town 05. Use the data as validation data.')
    parser.add_argument('--shuffle', default=False, help='shuffle the data order')
    parser.add_argument('--seed', default=3407, type=int) #42 #525
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')

    # VLM parameters
    parser.add_argument('--vlm_freeze', default='prompt_vlm', type=str, help='freeze vlm model')

    # Vision encoder parameters
    parser.add_argument('--encoder_freeze', default='vit_encoder', type=str, help='freeze part in encoder model')
    parser.add_argument('--n_layer', type=int, default=4, help='Number of transformer layers used in the transfuser')

    # Language model parameters
    parser.add_argument('--llm_freeze', default='llm_model', type=str, help='freeze language model')
    parser.add_argument('--qformer_freeze', default='vlm_adapter', type=str, help='freeze qformer model')
    parser.add_argument('--decoder_freeze', default='prompt_gpt', type=str, help='freeze part in decoder model')

    # prompt parameters
    parser.add_argument('--prompt_freeze', default='prompt_adapter', type=str, help='freeze part in encoder model')

    args = parser.parse_args()
    args.task_seq = ['town07', 'town04', 'town03']

    # Configure config
    config = GlobalConfig(root_dir=args.data_path, setting=args.setting)
    device = config.device #torch.device('cuda:{}'.format(local_rank))
    
    print('use device:', device)

    # set random seeds
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.set_device(device)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True # Wen want the highest performance

    # Create model and optimizers
    if config.model == 'GPT':
        model = init_gpt_vla(config)
    else:
        model = init_llama_vla(config)

    # Load Conitinual Learning Data
    data_loader = build_continual_dataloader(args, config)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print ('Total parameters: ', params)

    if args.load_pretrain:
        # Load checkpoint
        print("=============load=================")
        load_file = config.model_path
        model.load_state_dict(torch.load(load_file, map_location=torch.device('cpu'))["model"])

    model.cuda(device=device)

    start_time = time.time()

    trainer = Engine(model=model, data_loader=data_loader, args=args,
                     config=config, device=device, cur_epoch=args.start_epoch)

    train = args.train
    trainer.train_and_evaluate(train=train)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Total training time: {total_time_str}")

if __name__ == "__main__":
    
    mp.set_start_method('fork')
    print("Start method of multiprocessing:", mp.get_start_method())
    main()
