import os

class GlobalConfig:
    """ base architecture configurations """
	# Data
    seq_len = 4 # input timesteps
    pred_len = 6 # future waypoints planned
    start_task = 0 # the start task id for training
    num_tasks = 3 #number of tasks, 1 for joint training
    task_id = 3 #task id for close-loop evaluation
    device = 'cuda:0'
    image_type = 'segmentation'
    # image_type = 'rgb'
    label = 'path_planning'
    decision = 'dual' # longitudinal and lateral
    
    # Prompt Parameter
    pool_size = 9
    prompt_name = 'drive' #'l2p' #dual #coda #drive
    prompt_depth = 1 # 0 for l2p and 1 for l2p++
    e_prompt_len = 10 
    g_prompt_len = 10 
    top_k = 1 
    ortho = 1 # 1 for true, 0 for false
    attended_p = 1 # 1 for true, 0 for false
    
    ### GPTVLA #####
    # img_resolution = (224, 224) 

    # #GPTVLA
    # model = 'GPT' 
    # encoder = 'ViT-Tiny' #'CLIP-ViT-Large'
    # decoder = 'GPT2-xl' #'GPT2-xl'
    # mode = 'onboard' #onboard for bfloat16, otherwise float32

    # # GPTVLA Encoder
    # patch_size = 16 #14 
    # encoder_embed_dim = 192 #192 for vit_tiny_patch16_224
    # depth = 12 #12 for vit_tiny_patch16_224
    # num_heads = 3 # 3 for vit_tiny_patch16_224
    # ckpt_layer = 0
    # drop_path_rate = 0
    
    # # GPTVLA Decoder 
    # state_dim = 6 
    # act_dim = 9 # overall action space
    # embed_len = 3 
    # num_lat_classes = 6 # lateral action space
    # num_long_classes = 3 # longitudinal aciton space
    # num_regressions = 2
    # decoder_embed_dim = 1408
    # hidden_dim = 1600 #1600 for GPt2-XL 
    # max_emb_size = 4096
    # n_layer = 48 #48 for GPT2-XL
    # n_head = 25 #25 for GPT2-XL
    # dropout = 0.1

    #### MobileVLA ####
    img_resolution = (336, 336)
    
    ### MobileVLA
    model = 'LLaMA' 
    encoder = 'ViT-CLIP' 
    decoder = 'MobileLLaMA'
    mode = 'onboard' #onboard for bfloat16, otherwise float32

    #### MobileVLM Encoder #####
    patch_size = 16 #14
    encoder_embed_dim = 1024 #1024 for MobileVLM
    depth = 12 
    num_heads = 3 
    ckpt_layer = 0
    drop_path_rate = 0

    ### MobileVLM Decoder ####
    state_dim = 6 #10
    act_dim = 9 # overall action space 
    embed_len = 3 
    num_lat_classes = 6 # lateral action space
    num_long_classes = 3 # longitudinal aciton space
    num_regressions = 2
    decoder_embed_dim = 1408
    hidden_dim = 2048 #2048 for MobileVLM
    max_emb_size = 4096
    n_layer = 24 # 24 for MobileVLM
    n_head = 25
    dropout = 0.1
    
    # model_path = 'to be filled in'
    model_path = '/home/automan-apollo/Dropbox/seg_global_complex_language/drive_ViT-CLIP_MobileLLaMA_seed3407_fixlabelbug_fiximagebug_drama_poolingtoken_mobilevlm_coretemplate_mask2former_global_batch16_query16_town07_town04_town03_0.001_0.005_705050_10prompt_2024941462/town03_task3_final_checkpoint.pth' #

    ## wandb parameters
    dont_log_wandb = True
    project = "ProFounDrive-gptvla"
    wandb_group = "GPTVLA"

    # Domian Randomization
    scale = 1 # image pre-processing
    img_width = 320 # important this should be consistent with scale, e.g. scale = 1, img_width 320, scale=2, image_width 640
    augment = True
    inv_augment_prob = 0.1 # Probablity that data augmentation is applied is 1.0 - inv_augment_prob
    aug_max_rotation = 20 # degree
    
    # Controller
    turn_KP = 1.25
    turn_KI = 0.75
    turn_KD = 0.3
    turn_n = 40  # buffer size

    speed_KP = 5.0
    speed_KI = 0.5
    speed_KD = 1.0
    speed_n = 40  # buffer size

    # Carla
    weather = 'ClearNoon'
    max_throttle = 0.75  # upper limit on throttle signal value in dataset
    brake_speed = 0.1  # desired speed below which brake is triggered
    brake_ratio = 1.1  # ratio of speed to desired speed at which brake is triggered
    clip_delta = 0.35  # maximum change in speed input to logitudinal controller

    max_speed = 6.5
    collision_buffer = [2.5, 1.2]
    momentum = 0
    skip_frames = 2 #2
    detect_threshold = 0.04

    def __init__(self, root_dir='', setting='all', **kwargs):
        self.root_dir = root_dir
        if (setting == 'all'): # All towns used for training no validation data
            self.train_towns = os.listdir(self.root_dir)
            self.val_towns = [self.train_towns[0]]
            self.train_data, self.val_data = [], []
            for town in self.train_towns:
                root_files = os.listdir(os.path.join(self.root_dir, town)) #Town folders
                for file in root_files:
                    if not os.path.isfile(os.path.join(self.root_dir, file)):
                        self.train_data.append(os.path.join(self.root_dir, town, file))
            for town in self.val_towns:
                root_files = os.listdir(os.path.join(self.root_dir, town))
                for file in root_files:
                    if not os.path.isfile(os.path.join(self.root_dir, file)):
                        self.val_data.append(os.path.join(self.root_dir, town, file))

        elif (setting == '3-town'):
            print('Train on three different road structures: urban(Town3), highway(Town4), rural(Town7)')
            self.train_towns = os.listdir(self.root_dir)
            self.val_towns = self.train_towns
            self.train_data, self.val_data = [], []
            for town in self.train_towns:
                if ((town.find('town03') == -1) and (town.find('town04') == -1) and (town.find('town07') == -1)):  #We don't train on 05 and 02 to reserve them as test towns
                        continue
                root_files = os.listdir(os.path.join(self.root_dir, town)) #Town folders
                for file in root_files:
                    # if ((file.find('weather-0') == -1) and (file.find('weather-3') == -1) and (file.find('weather-6') == -1)):
                    if file.find('weather-0') == -1:
                        continue
                    subroot_files = os.listdir(os.path.join(self.root_dir, town, file))
                    for subfile in subroot_files:
                        if (subfile.find('data') == -1):
                            continue
                        if not os.path.isfile(os.path.join(self.root_dir, town, file, subfile)):
                            print("Train Folder: ", self.root_dir, town, file, subfile)
                            self.train_data.append(os.path.join(self.root_dir, town, file, subfile))

            for town in self.val_towns:
                if ((town.find('town03') == -1) and (town.find('town04') == -1) and (town.find('town07') == -1)):  #We don't train on 05 and 02 to reserve them as test towns
                        continue
                root_files = os.listdir(os.path.join(self.root_dir, town)) #Town folders
                for file in root_files:
                    # if ((file.find('weather-0') == -1) and (file.find('weather-3') == -1) and (file.find('weather-6') == -1)):
                    if file.find('weather-0') == -1:
                        continue
                    subroot_files = os.listdir(os.path.join(self.root_dir, town, file))
                    for subfile in subroot_files:
                        if (subfile.find('validation') == -1):
                            continue
                        if not os.path.isfile(os.path.join(self.root_dir, town, file, subfile)):
                            print("Validation Folder: ", self.root_dir, town, file, subfile)
                            self.val_data.append(os.path.join(self.root_dir, town, file, subfile))
                                  
            print(self.train_data, '\n', self.val_data, '\n')
        
        elif (setting == 'min-3-town'):
            print('Training mode with minimum town dataset')
            self.train_towns = os.listdir(self.root_dir)
            self.val_towns = self.train_towns
            self.train_data, self.val_data = [], []
            for town in self.train_towns:
                if ((town.find('town03') == -1) and (town.find('town04') == -1) and (town.find('town07') == -1)):  #We don't train on 05 and 02 to reserve them as test towns
                    continue
                # if ((town.find('town03') == -1)):  #We don't train on 05 and 02 to reserve them as test towns
                #     continue
                root_files = os.listdir(os.path.join(self.root_dir, town)) #Town folders
                for file in root_files:
                    if (file.find('weather-minimal') == -1):
                        continue
                    subroot_files = os.listdir(os.path.join(self.root_dir, town, file))
                    for subfile in subroot_files:
                        if (subfile.find('data') == -1):
                            continue
                        if not os.path.isfile(os.path.join(self.root_dir, town, file, subfile)):
                            print("Train Folder: ", self.root_dir, town, file, subfile)
                            self.train_data.append(os.path.join(self.root_dir, town, file, subfile))
                            
            for town in self.val_towns:
                if ((town.find('town03') == -1) and (town.find('town04') == -1) and (town.find('town07') == -1)):  #We don't train on 05 and 02 to reserve them as test towns
                    continue
                # if ((town.find('town03') == -1)):  #We don't train on 05 and 02 to reserve them as test towns
                #     continue
                root_files = os.listdir(os.path.join(self.root_dir, town)) #Town folders
                for file in root_files:
                    if (file.find('weather-minimal') == -1):
                        continue
                    subroot_files = os.listdir(os.path.join(self.root_dir, town, file))
                    for subfile in subroot_files:
                        if (subfile.find('validation') == -1):
                            continue
                        if not os.path.isfile(os.path.join(self.root_dir, town, file, subfile)):
                            print("Validation Folder: ", self.root_dir, town, file, subfile)
                            self.val_data.append(os.path.join(self.root_dir, town, file, subfile))

        elif (setting == 'eval'): #No training data needed during evaluation.
            pass
        else:
            print("Error: Selected setting: ", setting, " does not exist.")

        for k,v in kwargs.items():
            setattr(self, k, v)
