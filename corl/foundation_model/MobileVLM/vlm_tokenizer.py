import sys
import torch
import json
from PIL import Image
from pathlib import Path
import torch.nn as nn
import numpy as np
import sys
from .constants import LOGDIR, IMAGE_TOKEN_INDEX
from .conversation import conv_templates, SeparatorStyle
from .utils import disable_torch_init, process_images, tokenizer_image_token, KeywordsStoppingCriteria
from transformers import AutoTokenizer, BitsAndBytesConfig   

class VLMAdapter(nn.Module):
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        # json_file = open('./foundation_model/MobileVLM/template.json' , 'r') #../../corl/foundation_model/MobileVLM/template.json  # ./foundation_model/MobileVLM/template.json
        json_file = open('../../corl/foundation_model/MobileVLM/template.json' , 'r') #../../corl/foundation_model/MobileVLM/template.json  # ./foundation_model/MobileVLM/template.json
        self.instruction_temp = json.load(json_file)
        self.instruction_objects = self.instruction_temp['objects']
        self.instruction_non = self.instruction_temp['non-object']
        self.lat_instruction = self.instruction_temp['lat-instruction']
        self.long_instruction = self.instruction_temp['long-instrunction']
        self.general_instruction = self.instruction_temp['general']
        self.non_prompt = np.random.choice(self.instruction_non)
        self.object_prompt = np.random.choice(self.instruction_objects)
        self.prompt_list = None

    def generate_input_ids(self, prompt):
        input_ids = (tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").cuda())
        return input_ids

    def random_instruction(self):
        self.non_prompt = np.random.choice(self.instruction_non)
        self.object_prompt = np.random.choice(self.instruction_objects)
    
    def dynamic_prompt(self, vector, train):
        B, t, c = vector.shape
        vector = vector.reshape(-1, c)
        prompt_list = []
        input_ids_list = []
        for i in range(B*t):
            instruction = vector[i]
            target_waypoints = instruction[:2].data.cpu().numpy()
            target_waypoints = np.around(target_waypoints, decimals=2)
            vehicle_flag = instruction[2]
            bike_flag = instruction[3]
            ped_flag = instruction[4]
            red_light_flag = instruction[5]
            stop_flag = instruction[6]
            objects = ''
            num_objects = bike_flag + ped_flag + red_light_flag + stop_flag + vehicle_flag

            if train:
                if num_objects == 0:
                    prompt = np.random.choice(self.instruction_non)
                else:
                    prompt = np.random.choice(self.instruction_objects)
                    
                    if bike_flag:
                        objects += np.random.choice(['bikes, ', 'cyclists, '])
                        # objects += 'cyclists, '
                    if ped_flag:
                        objects += np.random.choice(['walkers, ', 'pedestrians '])
                        # objects += 'pedestrians, '
                    if red_light_flag:
                        objects += np.random.choice(['red light, ', 'red traffic light, '])
                        # objects += 'red light, '
                    if stop_flag:
                        objects += np.random.choice(['stop sign, '])
                        # objects += 'stop sign, '
                    if vehicle_flag:
                        objects += np.random.choice(['cars, ', 'vehicles, ', 'sedans, '])
                        # objects += 'cars, '
                    chunk_index = objects.rfind(',')
                    # if chunk_index != -1:
                    objects = objects[:chunk_index]
                    prompt = prompt.replace('objects', objects)
            else:
                if num_objects == 0:
                    prompt = np.random.choice(self.instruction_non)
                else:
                    prompt = np.random.choice(self.instruction_objects)

                    if bike_flag:
                        # objects += np.random.choice(['bikes, ', 'cyclists, '])
                        objects += 'cyclists, '
                    if ped_flag:
                        # objects += np.random.choice(['walkers, ', 'pedestrians '])
                        objects += 'pedestrians, '
                    if red_light_flag:
                        # objects += np.random.choice(['red light, ', 'red traffic light, '])
                        objects += 'red light, '
                    if stop_flag:
                        # objects += np.random.choice(['stop sign, '])
                        objects += 'stop sign, '
                    if vehicle_flag:
                        # objects += np.random.choice(['cars, ', 'vehicles, ', 'sedans, '])
                        objects += 'cars, '
                    chunk_index = objects.rfind(',')
                    # if chunk_index != -1:
                    objects = objects[:chunk_index]
                    prompt = prompt.replace('objects', objects)

            prompt = "<image>\n" + prompt
            prompt_list.append(prompt)
            input_ids = self.generate_input_ids(prompt)
            input_ids_list.append(input_ids)

        return input_ids_list, prompt_list
    
    def forward(self, input_dict, train):
        instruction_vector = input_dict['LanguageInstruction']        # [B, t, 6]
        input_ids, prompt_list = self.dynamic_prompt(instruction_vector, train)
        self.prompt_list = prompt_list
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        attention_masks = input_ids.ne(self.tokenizer.pad_token_id)
        return input_ids, attention_masks