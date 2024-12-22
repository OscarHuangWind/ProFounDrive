import os

class GlobalConfig:
    """ base architecture configurations """
	# Data
    seq_len = 4 # input timesteps
    # use different seq len for image and lidar
    img_seq_len = 1 
    lidar_seq_len = 1
    pred_len = 6 # future waypoints planned
    scale = 1 # image pre-processing
    img_width = 320 # important this should be consistent with scale, e.g. scale = 1, img_width 320, scale=2, image_width 640
    start_task = 0 # the start task id for training
    num_tasks = 3 #number of tasks, 1 for joint training
    task_id = 3 #task id for close-loop evaluation
    device = 'cuda:0'
    image_type = 'segmentation'
    # image_type = 'rgb'
    label = 'path_planning'
    decision = 'dual'
    
    # Prompt Parameter
    pool_size = 9
    prompt_name = 'drive' #'l2p' #dual #coda #drive
    prompt_depth = 1 # for l2p, 0 for l2p and 1 for l2p++
    e_prompt_len = 10 #2
    g_prompt_len = 10 #10
    top_k = 1 #3
    ortho = 1 # 1 for true, 0 for false
    attended_p = 1 # 1 for true, 0 for false
    
    ### GPTVLM #####
    img_resolution = (224, 224) 

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
    # embed_len = 3 #4 for foundation; 3 for vlm
    # num_lat_classes = 6 # lateral action space
    # num_long_classes = 3 # longitudinal aciton space
    # num_regressions = 2
    # decoder_embed_dim = 1408 #256
    # hidden_dim = 1600 #1600 for GPt2-XL 
    # max_emb_size = 4096
    # n_layer = 48 #48 for GPT2-XL
    # n_head = 25 #25 for GPT2-XL
    # dropout = 0.1

    #### MobileVLM ####
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
    embed_len = 3 #4 for foundation; 3 for vlm
    num_lat_classes = 6 # lateral action space
    num_long_classes = 3 # longitudinal aciton space
    num_regressions = 2
    decoder_embed_dim = 1408
    hidden_dim = 2048 #2048 for MobileVLM
    max_emb_size = 4096
    n_layer = 24 # 24 for MobileVLM
    n_head = 25
    dropout = 0.1
    
    # model_path = '/home/automan/Dropbox/InterFuser/corl/output/' + prompt_name + '_' + encoder + '_' + decoder + '_seed3407_20243922717/town04_task1_final_checkpoint.pth' #l2p 4-3-7
    # model_path = '/home/automan/Dropbox/InterFuser/corl/output/' + prompt_name + '_' + encoder + '_' + decoder + '_seed3407_20243922620/town03_task3_final_checkpoint.pth' #l2p 7-4-3
    # model_path = '/home/automan/Dropbox/InterFuser/corl/output/' + prompt_name + '_' + encoder + '_' + decoder + '_seed3407_20243922653/town03_task1_final_checkpoint.pth' #l2p 3-7-4    
    
    # model_path = '/home/automan/Dropbox/InterFuser/corl/output/' + prompt_name + '_' + encoder + '_' + decoder + '_2024210175/town04_task1_final_checkpoint.pth' #l2p++ 4-3-7
    # model_path = '/home/automan/Dropbox/InterFuser/corl/output/' + prompt_name + '_' + encoder + '_' + decoder + '_20243320258/town03_task1_final_checkpoint.pth' #l2p++ 3-7-4
    # model_path = '/home/automan/Dropbox/InterFuser/corl/output/' + prompt_name + '_' + encoder + '_' + decoder + '_20242123593/town03_task3_final_checkpoint.pth' #l2p++ 7-4-3

    # model_path = '/home/automan/Dropbox/InterFuser/corl/output/' + prompt_name + '_' + encoder + '_' + decoder + '_20242101716/town04_task1_final_checkpoint.pth' #dual 4-3-7
    # model_path = '/home/oscar/Dropbox/InterFuser/corl/output/' + prompt_name + '_' + encoder + '_' + decoder + '_202433194249/town03_task1_final_checkpoint.pth' #dual 3-7-4
    # model_path = '/home/automan/Dropbox/InterFuser/corl/output/' + prompt_name + '_' + encoder + '_' + decoder + '_202421235913/town03_task3_final_checkpoint.pth' #dual 7-4-3

    # model_path = '/home/automan/Dropbox/InterFuser/corl/output/' + prompt_name + '_' + encoder + '_' + decoder + '_20242101727/town04_task1_final_checkpoint.pth' #coda 4-3-7
    # model_path = '/home/automan/Dropbox/InterFuser/corl/output/' + prompt_name + '_' + encoder + '_' + decoder + '_202433194234/town03_task1_final_checkpoint.pth' #coda 3-7-4
    # model_path = '/home/automan/Dropbox/InterFuser/corl/output/' + prompt_name + '_' + encoder + '_' + decoder + '_202421235921/town03_task3_final_checkpoint.pth' #coda 7-4-3

    # model_path = '/home/automan/Dropbox/InterFuser/corl/output/' + prompt_name + '_' + encoder + '_' + decoder + '_2024120145756/town03_task1_final_checkpoint.pth'
    # model_path = '/home/oscar/Dropbox/InterFuser/corl/output/' + prompt_name + '_' + encoder + '_' + decoder + '_2024121152142/town03_task3_epoch10_acc92.7_checkpoint.pth'
    # model_path = '/home/automan/Dropbox/InterFuser/corl/output/' + prompt_name + '_' + encoder + '_' + decoder + '_2024122142031/town03_task3_epoch10_acc92.6_checkpoint.pth'
    # model_path = '/home/oscar/Dropbox/InterFuser/corl/output/' + prompt_name + '_' + encoder + '_' + decoder + '_2024121145225/town07_task3_epoch10_acc93.0_checkpoint.pth'
    # model_path = '/home/automan/Dropbox/InterFuser/corl/output/' + prompt_name + '_' + encoder + '_' + decoder + '_seed3407_orthogonal_202436161731/town03_task3_epoch10_acc94.1_checkpoint.pth'
    # model_path = '/home/oscar/Dropbox/InterFuser/corl/output/' + prompt_name + '_' + encoder + '_' + decoder + '_seed3407_bootstrap_orthogonal_202436165125/town03_task3_epoch15_acc94.6_checkpoint.pth' #7-4-3
    # model_path = '/home/automan/Dropbox/InterFuser/corl/output/' + prompt_name + '_' + encoder + '_' + decoder + '_seed3407_bootstrap_orthogonal_2024370474/town03_task1_final_checkpoint.pth' #3-7-4
    # model_path = '/home/oscar/Dropbox/InterFuser/corl/output/' + prompt_name + '_' + encoder + '_' + decoder + '_seed3407_bootstrap_orthogonal10_20243704826/town07_task3_epoch15_acc95.0_checkpoint.pth' #3-4-7

    # model_path = '/home/automan/Dropbox/InterFuser/corl/output/' + prompt_name + '_' + encoder + '_' + decoder + '_seed3407_20243513739/town03_task1_final_checkpoint.pth' #disjoint 3
    # model_path = '/home/automan/Dropbox/InterFuser/corl/output/' + prompt_name + '_' + encoder + '_' + decoder + '_seed3407_bootstrap_orthogonal_20243704637/town04_task1_final_checkpoint.pth' #disjoint 4
    # model_path = '/home/automan/Dropbox/InterFuser/corl/output/' + prompt_name + '_' + encoder + '_' + decoder + '_2024120145730/town07_task1_final_checkpoint.pth' #disjoint 7
    # model_path = '/home/automan/Dropbox/InterFuser/corl/output/' + prompt_name + '_' + encoder + '_' + decoder + '_2024121152142/town04_task1_final_checkpoint.pth' #disjoint 4

    # model_path = '/home/automan/Dropbox/InterFuser/corl/output/' + prompt_name + '_' + encoder + '_' + decoder + '_seed0_joint_20243917482/town07_task1_final_checkpoint.pth' #corl joint

    # pretrained gpt2
    # model_path = '/home/automan/Dropbox/InterFuser/corl/output/' + prompt_name + '_' + encoder + '_' + decoder + '_seed3407_pretrained_bootstrap_TuneGpt_NoTuneGru_global_local_key_orthogonal_2024317232527/town03_task1_final_checkpoint.pth' #3-4-7
    # model_path = '/home/automan/Dropbox/InterFuser/corl/output/' + prompt_name + '_' + encoder + '_' + decoder + '_seed3407_pretrained_bootstrap_TuneGpt_TuneGru_global_local_key_orthogonal_2024318225858/town04_task3_epoch5_acc95.3_checkpoint.pth' #3-7-4
    # model_path = '/home/automan/Dropbox/InterFuser/corl/output/' + prompt_name + '_' + encoder + '_' + decoder + '_seed3407_pretrained_Nobootstrap_TuneGpt_TuneGru_global_local_key_orthogonal_2024320225319/town03_task3_final_checkpoint.pth' #7-4-3
    # model_path = '/home/automan/Dropbox/InterFuser/corl/output/' + prompt_name + '_' + encoder + '_' + decoder + '_seed3407_pretrained_Nobootstrap_TuneGpt_TuneGru_global_local_key_orthogonal_2024320225510/town04_task3_epoch10_acc95.5_checkpoint.pth' #3-7-4
    # model_path = '/home/automan/Dropbox/InterFuser/corl/output/' + prompt_name + '_' + encoder + '_' + decoder + '_seed3407_pretrained_nobootstrap_1002020_TuneGpt_TuneGru_global_local_key_orthogonal_202432610423/town03_task3_final_checkpoint.pth' #7-4-3

    # model_path = '/home/users/ntu/songyan0/scratch/projects/VLMDrive-Pro/corl/output/' + prompt_name + '_' + encoder + '_' + decoder + '_seed3407_CLIPViTLarge_GPT2XL_seg_1.5s_local_pretrained_nobootstrap_0.001_0.001_1005050_10prompt_FreezeGpt_TuneGru_global_local_key_orthogonal_2024619131143/town03_task1_final_checkpoint.pth' #3-4-7
    # model_path = '/home/users/ntu/songyan0/scratch/projects/VLMDrive-Pro/corl/output/' + prompt_name + '_' + encoder + '_' + decoder + '_seed3407_CLIPViTLarge_GPT2XL_seg_1.5s_local_pretrained_nobootstrap_0.0005_0.0003_1005050_10prompt_FreezeGpt_TuneGru_global_local_key_orthogonal_20246191381/town03_task1_final_checkpoint.pth' #3-4-7
    # model_path = '/home/spyder/Dropbox/seg_3s_global_dual/' + prompt_name + '_' + encoder + '_' + decoder + '_seed3407_CLIPViTLarge_GPT2XL_dualdecision_rgb_1.5s_global_query16_pretrained_nobootstrap_0.0005_0.0003_1005050_10prompt_FreezeGpt_TuneGru_Fixglobal_local_key_orthogonal_2024621153049/town07_task3_final_checkpoint.pth' #3-4-7
    # model_path = '/media/spyder/94714162-4d32-4b72-b5d5-74dc17149d9c/wenhui_data/pretrained_models/dynamic_text_prompts/town07_task3_final_checkpoint.pth' #3-4-7

    # 理解语言的模型
    #'drive_vit_dt_seed3407_ViTTiny_GPT2_dualdecision_5detection_UniEmbeddings_Temporal_seg_1.5s_global_batch32_query16_347_10prompt_pretrained_nobootstrap_0.001_0.01_705050_10prompt_FreezeGpt_TuneGru_global_local_key_orthogonal_2024624183525'
    #'drive_vit_dt_seed3407_ViTTiny_GPT2_dualdecision_5detection_UniEmbeddings_Temporal_seg_1.5s_global_batch32_query16_374_10prompt_pretrained_nobootstrap_0.001_0.01_705050_10prompt_FreezeGpt_TuneGru_global_local_key_orthogonal_202462418380'
    # model_path = '/home/users/ntu/wenhui00/scratch/projects/VLMDrive-Pro/corl/language/' + prompt_name + '_' + encoder + '_' + decoder + '_seed3407_simpletemplate_seg_global_batch8_query16_town03_town04_town07_0.001_0.005_705050_10prompt_20247433042/town03_task1_final_checkpoint.pth' #3-4-7
    # model_path = '/home/users/ntu/wenhui00/scratch/projects/VLMDrive-Pro/corl/language/' + prompt_name + '_' + encoder + '_' + decoder + '_seed3407_seg_global_batch8_query16_town03_town04_town07_0.001_0.005_705050_10prompt_20247432542/town03_task1_final_checkpoint.pth' #3-4-7

    # all data
    # model_path = '/home/spyder/Dropbox/seg_global_complex_language/drive_ViT-Tiny_GPT2-xl_seed3407_alldata_coretemplate_seg_global_batch16_query16_town03_town04_town07_0.001_0.005_705050_10prompt_202471024617/town07_task3_final_checkpoint.pth' #3-4-7
    
    # nice model on argoverse
    # model_path = '/home/oscar/Dropbox/seg_global_complex_language/drive_ViT-Tiny_GPT2-xl_seed3407_coretemplate_seg_global_batch16_query16_town03_town04_town07_0.001_0.005_705050_10prompt_20247719949/town07_task3_final_checkpoint.pth' #cannot brake
    # model_path = '/home/oscar/Dropbox/seg_global_complex_language/drive_ViT-Tiny_GPT2-xl_seed3407_new_6route_mask2former_coretemplate_seg_global_batch16_query16_town03_town04_town07_0.001_0.005_705050_10prompt_2024712151117/town07_task3_final_checkpoint.pth' #bad
    # model_path = '/home/oscar/Dropbox/seg_global_complex_language/drive_ViT-Tiny_GPT2-xl_seed3407_new_6route_mask2former_coretemplate_seg_global_batch12_query16_town04_town03_town07_0.001_0.005_705050_10prompt_202471216937/town07_task3_final_checkpoint.pth' #left and keep lane
    # model_path = '/home/oscar/Dropbox/seg_global_complex_language/drive_ViT-Tiny_GPT2-xl_seed3407_new_6route_mask2former_coretemplate_seg_global_batch12_query16_town04_town07_town03_0.001_0.005_705050_10prompt_202471216937/town03_task3_final_checkpoint.pth' #bad
    # model_path = '/home/oscar/Dropbox/seg_global_complex_language/drive_ViT-Tiny_GPT2-xl_seed3407_new_6route_mask2former_coretemplate_seg_global_batch16_query16_town07_town04_town03_0.001_0.005_705050_10prompt_2024712151117/town03_task3_final_checkpoint.pth' #bad
    # model_path = '/home/oscar/Dropbox/seg_global_complex_language/drive_ViT-Tiny_GPT2-xl_seed3407_new_6route_mask2former_coretemplate_seg_global_batch16_query16_town07_town03_town04_0.001_0.005_705050_10prompt_2024712211752/town04_task3_final_checkpoint.pth' #traj not that good at left
    # model_path = '/home/oscar/Dropbox/seg_global_complex_language/drive_ViT-Tiny_GPT2-xl_seed3407_new_6route_mask2former_coretemplate_seg_global_batch16_query16_town03_town07_town04_0.001_0.005_705050_10prompt_202471221418/town04_task3_final_checkpoint.pth' #bad

    # long tail test model
    # model_path = '/home/oscar/Dropbox/seg_global_complex_language/drive_ViT-Tiny_GPT2-xl_seed3407_coretemplate_seg_global_batch16_query16_town03_town04_town07_0.001_0.005_705050_10prompt_20247719949/town07_task3_final_checkpoint.pth' #cannot brake
    # model_path = '/home/oscar/Dropbox/seg_global_complex_language/drive_ViT-Tiny_GPT2-xl_seed3407_new_6route_mask2former_coretemplate_seg_global_batch16_query16_town07_town04_town03_0.001_0.005_705050_10prompt_2024712151117/town07_task1_final_checkpoint.pth' #
    # model_path = '/home/oscar/Dropbox/seg_global_complex_language/NoPrompt_ViT-Tiny_GPT2-xl_seed3407_resettest_coretemplate_mask2former_global_batch16_query16_town03_town04_town07_0.001_0.005_705050_10prompt_2024729213042/town04_task2_epoch5_acc63.5_checkpoint.pth' #
    # model_path = '/home/oscar/Dropbox/seg_global_complex_language/NoPrompt_ViT-Tiny_GPT2-xl_seed3407_onboard_continue_llmfreeze_coretemplate_mask2former_global_batch26_query16_town04_town07_town03_0.001_0.005_705050_10prompt_2024813164524/town03_task3_final_checkpoint.pth' #
    # model_path = '/data2/pretrained-models/MobileVLM-1.7B'
    # generalization test model
    # model_path = '/home/oscar/Dropbox/seg_global_complex_language/drive_ViT-Tiny_GPT2-xl_seed3407_new_6route_mask2former_coretemplate_seg_global_batch16_query16_town07_town04_town03_0.001_0.005_705050_10prompt_2024712151117/town03_task3_final_checkpoint.pth' #keep lane during the right

    # Driving Score model
    # model_path = '/home/oscar/Dropbox/seg_global_complex_language/drive_ViT-Tiny_GPT2-xl_seed3407_new_6route_mask2former_coretemplate_seg_global_batch16_query16_town07_town03_town04_0.001_0.005_705050_10prompt_2024712211752/town04_task3_final_checkpoint.pth' # 
    
    # GptVLA
    # model_path = '/home/oscar/Dropbox/seg_global_complex_language/drive_ViT-Tiny_GPT2-xl_seed3407_new_6route_mask2former_coretemplate_seg_global_batch16_query16_town07_town04_town03_0.001_0.005_705050_10prompt_2024712151117/town03_task3_final_checkpoint.pth' #
    
    # MobileVLA
    # model_path = '/home/oscar/Dropbox/seg_global_complex_language/drive_ViT-CLIP_MobileLLaMA_seed3407_drama_poolingtoken_mobilevlm_coretemplate_mask2former_global_batch16_query16_town07_town04_town03_0.001_0.005_705050_10prompt_202482420856/town03_task3_final_checkpoint.pth' #
    model_path = '/home/oscar/Dropbox/seg_global_complex_language/drive_ViT-CLIP_MobileLLaMA_seed3407_fixlabelbug_fiximagebug_drama_poolingtoken_mobilevlm_coretemplate_mask2former_global_batch16_query16_town07_town04_town03_0.001_0.005_705050_10prompt_2024941462/town03_task3_final_checkpoint.pth' #

    # Decision Transformer
    # model_path = '/home/oscar/Dropbox/seg_global_complex_language/NoPrompt_ViT-Tiny_GPT2-xl_seed3407_onboard_continue_llmfreeze_coretemplate_mask2former_global_batch26_query16_town04_town07_town03_0.001_0.005_705050_10prompt_2024813164524/town03_task3_final_checkpoint.pth' #
    # model_path = '/home/oscar/Dropbox/seg_global_complex_language/NoPrompt_ViT-CLIP_MobileLLaMA_seed3407_fixlabelbug_mobilevla_coretemplate_mask2former_global_batch16_query16_town07_town04_town03_0.001_0.005_705050_10prompt_2024910183839/town03_task3_final_checkpoint.pth' #

    # Joint Training
    # model_path = '/home/oscar/Dropbox/seg_global_complex_language/drive_ViT-Tiny_GPT2-xl_seed3407_joint_fixlabelbug_gptvla_coretemplate_mask2former_global_batch16_query16_town07_town04_town03_0.001_0.005_705050_10prompt_202497123134/town07_task1_final_checkpoint.pth' #

    # finetuned MobileVLA
    # model_path = '/home/oscar/Dropbox/VisionFoundationVehicle/corl/realworld_evaluate/finetune/drive_ViT-CLIP_MobileLLaMA_seed3407_15_finetune_drama_poolingtoken_mobilevlm_coretemplate_mask2former_global_batch4_query16_CoR_0.0001_0.001_10prompt_202482711218/CoR_task1_final_checkpoint.pth'

    ## wandb parameters
    dont_log_wandb = True
    # wandb_entity = "OscarHuang"
    # wandb_entity = "spyderzsy"
    project = "ProFounDrive-mobilevla"
    wandb_group = "MobileVLA"

    # Domian Randomization
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

    max_speed = 5
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

        elif (setting == '02_05_withheld'): #Town02 and 05 withheld during training
            print("Skip Town02 and Town05")
            self.train_towns = os.listdir(self.root_dir) #Scenario Folders
            self.val_towns = self.train_towns # Town 02 and 05 get selected automatically below
            self.train_data, self.val_data = [], []
            for town in self.train_towns:
                root_files = os.listdir(os.path.join(self.root_dir, town)) #Town folders
                for file in root_files:
                    if ((file.find('Town02') != -1) or (file.find('Town05') != -1)):  #We don't train on 05 and 02 to reserve them as test towns
                        continue
                    if not os.path.isfile(os.path.join(self.root_dir, file)):
                        print("Train Folder: ", file)
                        self.train_data.append(os.path.join(self.root_dir, town, file))
            for town in self.val_towns:
                root_files = os.listdir(os.path.join(self.root_dir, town))
                for file in root_files:
                    if ((file.find('Town02') == -1) and (file.find('Town05') == -1)): # Only use Town 02 and 05 for validation
                        continue
                    if not os.path.isfile(os.path.join(self.root_dir, file)):
                        print("Val Folder: ", file)
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
