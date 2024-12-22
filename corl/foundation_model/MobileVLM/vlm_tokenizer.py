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

def load_pretrained_model(model_path, load_8bit=False, load_4bit=False, device_map="auto", device="cuda"):

    from mobilellama import MobileLlamaForCausalLM

    kwargs = {"device_map": device_map}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = MobileLlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
    if mm_use_im_patch_token:
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))

    vision_tower = model.get_vision_tower()
    if 'v2' in getattr(model.config, "mm_projector_type", "ldpnet"):
        vision_tower.load_image_processor()
    elif not vision_tower.is_loaded:
        vision_tower.load_model()
    vision_tower.to(device=device, dtype=torch.float16)
    image_processor = vision_tower.image_processor

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048
    
    return tokenizer, model, image_processor, context_len    

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
        # json_file = open('../../corl/foundation_model/decoder/template.json', 'r')
        # self.instruction_temp = json.load(json_file)
        # instruction_objects = self.instruction_temp['objects']
        # instruction_non = self.instruction_temp['non-object']
        # lat_instruction = self.instruction_temp['lat-instruction']
        # long_instruction = self.instruction_temp['long-instrunction']
        # general_instruction = self.instruction_temp['general']
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

                # if num_objects == 0:
                #     prompt = self.non_prompt
                # else:
                #     prompt = self.object_prompt

                # if bike_flag:
                #     objects += 'cyclists, '
                # if ped_flag:
                #     objects += 'pedestrians, '
                # if red_light_flag:
                #     objects += 'red light, '
                # if stop_flag:
                #     objects += 'stop sign, '
                # if vehicle_flag:
                #     objects += 'cars, '
                # chunk_index = objects.rfind(',')
                # objects = objects[:chunk_index]
                # prompt = prompt.replace('objects', objects)

            # prompt = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: <image>\n" +\
            #         prompt + ' ASSISTANT:'
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


def attach_debugger():
    import debugpy
    debugpy.listen(5678)
    print("Waiting for debugger!")
    debugpy.wait_for_client()
    print("Attached!")


def test_batch_prompot():
    from transformers import GPT2Tokenizer, GPT2LMHeadModel

    import torch
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.add_special_tokens({'bos_token': '</s>'})
    tokenizer.add_special_tokens({'eos_token': '</s>'})
    tokenizer.add_special_tokens({'unk_token': '</s>'})
    # llm_model = GPT2Model.from_pretrained('gpt2', torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).cuda()
    # sentences = [
    #     "The cat sat on the mat.",
    #     "Once upon a time in a land far, far away,",
    #     # "To be or not to be, that is the question:"
    # ]
    
    # encoded_inputs = tokenizer(sentences, return_tensors='pt', padding='longest', truncation=True, max_length=64).to('cuda')
    # import pdb; pdb.set_trace()
    # print("Encoded inputs:", encoded_inputs)
    # text_embeds = llm_model.get_input_embeddings()(encoded_inputs.input_ids)

    # model = GPT2LMHeadModel.from_pretrained('gpt2')

    # 准备输入数据
    # model = GPT2LMHeadModel.from_pretrained('gpt2')

    # sentences = [
    #     "The cat sat on the mat.",
    #     "Once upon a time in a land far, far away,",
    #     "To be or not to be, that is the question:"
    # ]

    # # 对句子进行编码，并添加填充
    # encoded_inputs = tokenizer(sentences, return_tensors='pt', padding=True)

    # # 打印编码后的输入
    # print("Encoded inputs:", encoded_inputs)

    # # 获取模型的嵌入层
    # embedding_layer = model.get_input_embeddings()

    # # 将编码的输入ID转换为嵌入向量
    # input_embeddings = embedding_layer(encoded_inputs['input_ids'])

    # # 打印嵌入向量的形状
    # print("Input embeddings shape:", input_embeddings.shape)
    # import pdb; pdb.set_trace()


    # 加载GPT-2模型和分词器
    # model = GPT2LMHeadModel.from_pretrained('gpt2')

    # # 准备输入数据
    # sentences = [
    #     "The cat sat on the mat.",
    #     "Once upon a time in a land far, far away,",
    #     "To be or not to be, that is the question:"
    # ]

    # # 对句子进行编码，并添加填充
    # encoded_inputs = tokenizer(sentences, return_tensors='pt', padding=True)

    # # 打印编码后的输入
    # print("Encoded inputs:", encoded_inputs)

    # # 获取模型的嵌入层
    # embedding_layer = model.get_input_embeddings()

    # # 将编码的输入ID转换为嵌入向量
    # input_embeddings = embedding_layer(encoded_inputs['input_ids'])

    # # 打印嵌入向量的形状
    # print("Input embeddings shape:", input_embeddings.shape)


if __name__ == '__main__':
    # attach_debugger()

    # test QFormer
    test_input_data = torch.randn(1, 2, 20, 1408).cuda()
    prompt = torch.randint(0, 2, (1, 2, 5)).cuda()
    waypoints = torch.randn((1, 2, 2)).cuda()
    prompt = torch.cat((waypoints, prompt), dim=-1)

    input_dict = {
        'ImageTokenEmbedding': test_input_data,
        'LanguageInstruction': prompt
    }
    # import pdb; pdb.set_trace()
    model_path = "mtgv/MobileVLM_V2-1.7B" 
    tokenizer, llm_model, image_processor, _ = load_pretrained_model(model_path, False, False)
    adapter = VLMAdapter(tokenizer).cuda()
    hidden_state, mask = adapter(input_dict, llm_model.get_input_embeddings())



    