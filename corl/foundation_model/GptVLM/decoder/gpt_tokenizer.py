from transformers import GPT2Tokenizer, GPT2Model
import torch
import torch.nn as nn

# from lavis.models.blip2_models.Qformer import BertConfig
from transformers import PretrainedConfig
import contextlib
from transformers import Blip2Model
import json
import numpy as np

seed = 3407 #42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True # Wen want the highest performance

class VLMAdapter(nn.Module):
    def __init__(self, max_txt_len=128, num_token_queries=16, llm_hidden_dim=768):
        super().__init__()
        # load tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.max_txt_len = max_txt_len
        self.qformer, self.query_tokens = self.init_Qformer_BLIP2(num_token_queries)
        self.llm_proj = nn.Linear(self.qformer.config.hidden_size, llm_hidden_dim)
        self.llm_hidden_dim = llm_hidden_dim
        self.tokenizer.pad_token = self.tokenizer.eos_token 
        self.prompt_list = None

        json_file = open('../../corl/foundation_model/GptVLM/decoder/template.json', 'r') #'./foundation_model/decoder/template.json'
        self.instruction_temp = json.load(json_file)
        self.instruction_objects = self.instruction_temp['objects']
        self.instruction_non = self.instruction_temp['non-object']
        self.lat_instruction = self.instruction_temp['lat-instruction']
        self.long_instruction = self.instruction_temp['long-instrunction']
        self.general_instruction = self.instruction_temp['general']
        self.non_prompt = np.random.choice(self.instruction_non)
        self.object_prompt = np.random.choice(self.instruction_objects)

    def random_instruction(self):
        self.non_prompt = np.random.choice(self.instruction_non)
        self.object_prompt = np.random.choice(self.instruction_objects)

    def maybe_autocast(self, dtype=torch.float16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        # enable_autocast = self.device != torch.device("cpu")
        enable_autocast = True

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()
    
    def init_Qformer_BLIP2(self, num_query_token):
        Qformer = Blip2Model.from_pretrained('salesforce/blip2-opt-2.7b', low_cpu_mem_usage=True).qformer
        qformer_config = Qformer.config
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, qformer_config.hidden_size)
        )
        return Qformer, query_tokens
    
    def dynamic_prompt(self, vector, train):
        B, t, c = vector.shape
        vector = vector.reshape(-1, c)
        prompt_list = []
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
            prompt_list.append(prompt)
        return prompt_list
        
    
    def transfer_vector_to_prompt(self, vector):
        # convert the provided instruction vector to a fixed prompt template
        B, t, c = vector.shape
        vector = vector.reshape(-1, c)
        prompt_list = []
        for i in range(B*t):
            instruction = vector[i]
            target_waypoints = instruction[:2].data.cpu().numpy()
            target_waypoints = np.around(target_waypoints, decimals=1)
            fg_flag = instruction[2:].sum()
            instruction[2:] = instruction[2:]

            if fg_flag:
                vehicle_flag = instruction[2]
                bike_flag = instruction[3]
                ped_flag = instruction[4]
                red_light_flag = instruction[5]
                stop_flag = instruction[6]
                if red_light_flag and fg_flag == 1:
                    # only red light is detected
                    prompt = 'There is a red light,'
                else:
                    prompt = 'There are'
                    if vehicle_flag:
                        prompt += ' vehicles,'
                    if bike_flag:
                        prompt += ' bikes,'
                    if ped_flag:
                        prompt += ' pedestrians,'
                    
                    if red_light_flag:
                        prompt += ' a red light,'
                    
                    if stop_flag:
                        prompt += ' a stop sign,'
                
                prompt += f' and the target waypoint is {target_waypoints[0], target_waypoints[1]}.'

            else:
                # no foreground objects detected
                prompt = f'The target waypoint is {target_waypoints[0], target_waypoints[1]}.'

            prompt_list.append(prompt)

        return prompt_list
    
    def forward(self, input_dict, llm_embed_func, train):
        instruction_vector = input_dict['LanguageInstruction']        # [B, t, 6]
        input_embeds = input_dict['ImageTokenEmbedding']   # [B, t, n, c]
        B, t, n, c = input_embeds.shape
        input_embeds = input_embeds.view(B*t, n, c)
        device = input_embeds.device

        # convert the instruction vector to prompt
        prompt_list = self.dynamic_prompt(instruction_vector, train)
        self.prompt_list = prompt_list

        # load QFormer for visual feature extraction
        image_attention_mask = torch.ones(input_embeds.size()[:-1], dtype=torch.long, device=input_embeds.device)

        query_tokens = self.query_tokens.expand(input_embeds.shape[0], -1, -1)
        query_outputs = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=input_embeds,
            encoder_attention_mask=image_attention_mask,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=True,
        )
        query_output = query_outputs[0]
        multimodal_inputs = self.llm_proj(query_output)
        # tokenizer the prompts and obtain the embeddings
        self.tokenizer.padding_side = "right"
        self.tokenizer.truncation_side = 'left'
        text_input_tokens = self.tokenizer(
            prompt_list,
            padding='longest',
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(device)
                
        text_input_embeds = llm_embed_func(text_input_tokens.input_ids)
        llm_inputs = torch.cat((multimodal_inputs, text_input_embeds), dim=1)
        multimodal_attention_mask = torch.ones(*multimodal_inputs.shape[:2]).to(device)
        llm_attention_mask = torch.cat((multimodal_attention_mask, text_input_tokens.attention_mask), dim=1)
        # reshape the llm_input to original shape
        llm_inputs = llm_inputs.view(B, t, -1, self.llm_hidden_dim)
        llm_attention_mask = llm_attention_mask.view(B, t, -1)
        return llm_inputs, llm_attention_mask