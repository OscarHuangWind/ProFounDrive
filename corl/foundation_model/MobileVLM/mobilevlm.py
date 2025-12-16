import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from transformers import AutoTokenizer, BitsAndBytesConfig
from .vision_encoder import build_vision_tower
from .vision_projector import build_vision_projector
from .vlm_tokenizer import VLMAdapter
from .constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, \
    DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

class MobileVLMMetaModel:

    def __init__(self, config):
        super(MobileVLMMetaModel, self).__init__(config)
        if hasattr(config, "mm_vision_tower"):
            config.mm_vision_select_feature = 'cls_patch'
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.mm_projector = build_vision_projector(config)

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower
    
    def get_vision_projector(self):
        vision_projector = getattr(self, 'mm_projector', None)
        if type(vision_projector) is list:
            vision_projector = vision_projector[0]
        return vision_projector

    def initialize_vision_modules(self, model_args, fsdp=None):
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        self.config.mm_vision_tower = model_args.vision_tower
        vision_tower = model_args.vision_tower

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        elif self.get_vision_tower().vision_tower_name != vision_tower:
            vision_tower = build_vision_tower(model_args)
            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
                vision_tower.load_model()
            else:
                vision_tower = self.vision_tower
                vision_tower.load_model()

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        # Build VisionTower
        vision_tower = build_vision_tower(model_args)
        if fsdp is not None and len(fsdp) > 0:
            self.vision_tower = [vision_tower]
        else:
            self.vision_tower = vision_tower
        self.config.mm_hidden_size = vision_tower.hidden_size
        # Build Vision-Projector
        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_vision_projector(self.config)
        # In case it is frozen by LoRA
        for p in self.mm_projector.parameters():
            p.requires_grad = True
        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}
            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))


class MobileVLMMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()
    
    def get_vision_projector(self):
        return self.get_model().get_vision_projector()

    def encode_images(self, images):
        image_features = self.get_model().get_vision_tower()(images)
        # this is a hard code here
        if image_features.shape[1] == 577:
            q, image_features = image_features[:, 0], image_features[:, 1:]
            image_features = self.get_model().mm_projector(image_features)      # q in shape [B*t, dim] and image_features in shape [B*t, token_num, dim]
            return [q, image_features]
        else:
            image_features = self.get_model().mm_projector(image_features)
            return [image_features]

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, attention_mask, past_key_values=None, labels=None, images=None, has_q=False
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            if past_key_values is not None and vision_tower is not None and images is not None and input_ids.shape[1] == 1:
                attention_mask = torch.ones((attention_mask.shape[0], past_key_values[-1][-1].shape[-2] + 1), dtype=attention_mask.dtype, device=attention_mask.device)
            return input_ids, attention_mask, past_key_values, None, labels
        
        if type(images) is list or images.ndim == 5:
            concat_images = torch.cat([image for image in images], dim=0)
            image_features = self.encode_images(concat_images)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            image_features = [x.flatten(0, 1) for x in image_features]
        else:
            return_data = self.encode_images(images)
            if len(return_data) == 2:
                prompt_queries, image_features = return_data
            else:
                image_features = return_data[0]
        
        # NOTE: this is a hard code here, we extract a fixed number of visual tokens for efficient computation
        pooling_layer = nn.AdaptiveAvgPool1d(20)
        image_features = image_features.permute(0, 2, 1)
        image_features = pooling_layer(image_features).permute(0, 2, 1)

        new_input_embeds = []
        new_labels = [] if labels is not None else None
        cur_image_idx = 0

        for batch_idx, cur_input_ids in enumerate(input_ids):
            if (cur_input_ids == IMAGE_TOKEN_INDEX).sum() == 0:
                # multimodal LLM, but the current sample is not multimodal
                # FIXME: this is a hacky fix, for deepspeed zero3 to work
                half_len = cur_input_ids.shape[0] // 2
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids[:half_len])
                cur_input_embeds_2 = self.get_model().embed_tokens(cur_input_ids[half_len:])
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0], cur_input_embeds_2], dim=0)
                new_input_embeds.append(cur_input_embeds)
                if labels is not None:
                    new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue
            image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            cur_new_input_embeds = []
            if labels is not None:
                cur_labels = labels[batch_idx]
                cur_new_labels = []
                assert cur_labels.shape == cur_input_ids.shape
            while image_token_indices.numel() > 0:
                cur_image_features = image_features[cur_image_idx]
                image_token_start = image_token_indices[0]
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:image_token_start-1]).detach())
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[image_token_start-1:image_token_start]))
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[image_token_start+1:image_token_start+2]))
                    if labels is not None:
                        cur_new_labels.append(cur_labels[:image_token_start])
                        cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
                        cur_new_labels.append(cur_labels[image_token_start:image_token_start+1])
                        cur_labels = cur_labels[image_token_start+2:]
                else:
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:image_token_start]))
                    cur_new_input_embeds.append(cur_image_features)
                    if labels is not None:
                        cur_new_labels.append(cur_labels[:image_token_start])
                        cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
                        cur_labels = cur_labels[image_token_start+1:]
                cur_image_idx += 1
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
                    cur_input_ids = cur_input_ids[image_token_start+2:]
                else:
                    cur_input_ids = cur_input_ids[image_token_start+1:]
                image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            if cur_input_ids.numel() > 0:
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids).detach())
                else:
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids))
                if labels is not None:
                    cur_new_labels.append(cur_labels)
            cur_new_input_embeds = [x.to(device=self.device) for x in cur_new_input_embeds]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
            new_input_embeds.append(cur_new_input_embeds)
            if labels is not None:
                cur_new_labels = torch.cat(cur_new_labels, dim=0)
                new_labels.append(cur_new_labels)

        if any(x.shape != new_input_embeds[0].shape for x in new_input_embeds):
            max_len = max(x.shape[0] for x in new_input_embeds)

            new_input_embeds_align = []
            for cur_new_embed in new_input_embeds:
                cur_new_embed = torch.cat((cur_new_embed, torch.zeros((max_len - cur_new_embed.shape[0], cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0)
                new_input_embeds_align.append(cur_new_embed)
            new_input_embeds = torch.stack(new_input_embeds_align, dim=0)

            if labels is not None:
                new_labels_align = []
                _new_labels = new_labels
                for cur_new_label in new_labels:
                    cur_new_label = torch.cat((cur_new_label, torch.full((max_len - cur_new_label.shape[0],), IGNORE_INDEX, dtype=cur_new_label.dtype, device=cur_new_label.device)), dim=0)
                    new_labels_align.append(cur_new_label)
                new_labels = torch.stack(new_labels_align, dim=0)

            if attention_mask is not None:
                new_attention_mask = []
                for cur_attention_mask, cur_new_labels, cur_new_labels_align in zip(attention_mask, _new_labels, new_labels):
                    new_attn_mask_pad_left = torch.full((cur_new_labels.shape[0] - labels.shape[1],), True, dtype=attention_mask.dtype, device=attention_mask.device)
                    new_attn_mask_pad_right = torch.full((cur_new_labels_align.shape[0] - cur_new_labels.shape[0],), False, dtype=attention_mask.dtype, device=attention_mask.device)
                    cur_new_attention_mask = torch.cat((new_attn_mask_pad_left, cur_attention_mask, new_attn_mask_pad_right), dim=0)
                    new_attention_mask.append(cur_new_attention_mask)
                attention_mask = torch.stack(new_attention_mask, dim=0)
                assert attention_mask.shape == new_labels.shape
        else:
            new_input_embeds = torch.stack(new_input_embeds, dim=0)
            if labels is not None:
                new_labels  = torch.stack(new_labels, dim=0)

            if attention_mask is not None:
                new_attn_mask_pad_left = torch.full((attention_mask.shape[0], new_input_embeds.shape[1] - input_ids.shape[1]), True, dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat((new_attn_mask_pad_left, attention_mask), dim=1)
                assert attention_mask.shape == new_input_embeds.shape[:2]
        if not has_q:
            return new_input_embeds, attention_mask, None
            # return None, attention_mask, past_key_values, new_input_embeds, new_labels
        else:
            return new_input_embeds, attention_mask, prompt_queries

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

class PromptVLM(nn.Module):
    def __init__(
            self,
            config,
            state_dim,
            act_dim,
            hidden_size,
            seq_len=None,
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
        self.prompt_len = prompt_len
        self.hidden_size = hidden_size
        self.img_resolution = 224
        self.img_state_dim = self.img_resolution**2 * 3
        
        ####### replace this model path with yours #########
        # model_path = "mtgv/MobileVLM_V2-1.7B" 
        model_path = '/home/automan-apollo/Dropbox/VisionFoundationVehicle/corl/realworld_evaluate/mobilevlm_v2-2.finetune'
        self.tokenizer, self.llm_model, self.image_processor, _ = self.load_pretrained_model(model_path, False, False)
        self.vlm_adapter = VLMAdapter(self.tokenizer)
        
        if self.config.mode == 'onboard':

            self.embed_vlm_state = torch.nn.Linear(self.hidden_size, self.hidden_size, dtype=torch.bfloat16)
            self.embed_image_state = torch.nn.Linear(self.hidden_size, self.hidden_size, dtype=torch.bfloat16)
            self.embed_goal_state = torch.nn.Linear(self.config.num_regressions, self.hidden_size, dtype=torch.bfloat16)
            self.embed_return = torch.nn.Linear(1, self.hidden_size, dtype=torch.bfloat16)
            self.embed_action = torch.nn.Linear(self.act_dim, self.hidden_size, dtype=torch.bfloat16)
            self.embed_ln = nn.LayerNorm(self.hidden_size, dtype=torch.bfloat16)
            self.embed_timestep = nn.Embedding(max_emb_size, self.hidden_size, dtype=torch.bfloat16)

            self.predict_class = nn.Sequential(
                *(
                    [nn.Linear(self.hidden_size, self.act_dim, dtype=torch.bfloat16)] + 
                    ([nn.Softmax(dim=2)] if softmax else []))
            )

        else:

            self.embed_vlm_state = torch.nn.Linear(self.hidden_size, self.hidden_size, dtype=torch.float16)
            self.embed_image_state = torch.nn.Linear(self.hidden_size, self.hidden_size, dtype=torch.float16)
            self.embed_goal_state = torch.nn.Linear(self.config.num_regressions, self.hidden_size, dtype=torch.float16)
            self.embed_return = torch.nn.Linear(1, self.hidden_size, dtype=torch.float16)
            self.embed_action = torch.nn.Linear(self.act_dim, self.hidden_size, dtype=torch.float16)
            self.embed_ln = nn.LayerNorm(self.hidden_size, dtype=torch.float16)
            self.embed_timestep = nn.Embedding(max_emb_size, self.hidden_size, dtype=torch.float16)
    
            self.predict_class = nn.Sequential(
                *(
                    [nn.Linear(self.hidden_size, self.act_dim, dtype=torch.float16)] + 
                    ([nn.Softmax(dim=2)] if softmax else []))
            )

    def load_pretrained_model(self, model_path, load_8bit=False, load_4bit=False, device_map={"": 0}, device="cuda"):

        from .mobilellama import MobileLlamaForCausalLM

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

    def forward(self, states, actions, returns_to_go, timesteps,
                detections, decisions, target_waypoints, attention_mask=None,
                prompt=None, q=None, train=None, task_id=None):
        batch_size, seq_length = actions.shape[0], actions.shape[1]
        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length)).to(actions.device)

        # embed each modality with a different head
        if decisions is None:
            condition_states = torch.cat((target_waypoints, detections), dim=-1)
        elif target_waypoints is None   : 
            condition_states = torch.cat((decisions, detections), dim=-1)
                
        '''
        Add Mobilevlm tokenizer for instruction tokens
        '''
        input_dicts = {'LanguageInstruction': condition_states}
        input_ids, llm_attention_mask = self.vlm_adapter(input_dicts, train=train)       # [batch*seq, -1, hidden_dim]

        if prompt is not None:
            prompt_query = True
        else:
            prompt_query = False

        vlm_states, vlm_attention_mask, q = self.llm_model.prepare_inputs_labels_for_multimodal(input_ids, llm_attention_mask, images=states, has_q=prompt_query)

        vlm_states = vlm_states.view(batch_size, seq_length, -1, self.hidden_size)
        vlm_attention_mask = vlm_attention_mask.view(batch_size, seq_length, -1)

        rtg_embeddings = self.embed_return(returns_to_go).unsqueeze(2)
        vlm_embeddings = self.embed_vlm_state(vlm_states)
        goal_embeddings = self.embed_goal_state(target_waypoints).unsqueeze(2)
        state_embeddings = torch.cat((goal_embeddings, vlm_embeddings), dim=-2)
        action_embeddings = self.embed_action(actions).unsqueeze(2)
        time_embeddings = self.embed_timestep(timesteps.reshape(-1, self.seq_len)).unsqueeze(2)

        action_embeddings = action_embeddings + time_embeddings
        rtg_embeddings = rtg_embeddings + time_embeddings
        state_embeddings = state_embeddings + time_embeddings

        llm_embeddings = torch.cat((rtg_embeddings, state_embeddings, action_embeddings), dim=-2)
        llm_inputs = self.embed_ln(llm_embeddings).view(batch_size, -1, self.hidden_size)

        rtg_attention_mask = attention_mask.unsqueeze(-1).repeat(1, 1, rtg_embeddings.shape[-2])
        goal_attention_mask = attention_mask.unsqueeze(-1).repeat(1, 1, goal_embeddings.shape[-2])
        state_attention_mask = torch.cat((goal_attention_mask, vlm_attention_mask), dim=-1)
        action_attention_mask = attention_mask.unsqueeze(-1).repeat(1, 1, action_embeddings.shape[-2])
        
        if prompt is not None:
            prompt_len = int(self.prompt_len / 2) # half for key and value
            prompt_attention_mask = torch.ones((batch_size, seq_length, prompt_len)).to(actions.device)

            llm_attention_mask = torch.cat((prompt_attention_mask,
                                            rtg_attention_mask,
                                            state_attention_mask,
                                            action_attention_mask), dim=-1).view(batch_size, -1)
        else:
            llm_attention_mask = torch.cat((rtg_attention_mask,
                                            state_attention_mask,
                                            action_attention_mask), dim=-1).view(batch_size, -1)            

        hidden_states, prompt_loss = self.llm_model(
                input_ids=None,
                inputs_embeds=llm_inputs,
                attention_mask=llm_attention_mask,
                output_hidden_states=True,
                return_dict=True,
                prompt=prompt,
                q=q,
                train=train,
                task_id=task_id,
            )

        hidden_states = hidden_states.reshape(batch_size, seq_length, -1, self.hidden_size).permute(0, 2, 1, 3)
        action_preds = self.predict_class(hidden_states[:,-2])[:, -seq_length:, :]  # predict next action given state

        if torch.any(torch.isnan(action_preds)):
            print('lat wtf?')

        return action_preds, prompt_loss, hidden_states[:,-2]

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
            attention_mask = attention_mask.to(dtype=torch.float16, device=timesteps.device).reshape(1, -1)

            actions = torch.cat(
                [torch.zeros((actions.shape[0], 
                              self.seq_len - actions.shape[1], 
                              self.act_dim),
                              dtype=torch.float16,
                              device=actions.device), actions], dim=1)
           
            returns_to_go = torch.cat(
                [torch.zeros((returns_to_go.shape[0], 
                              self.seq_len-returns_to_go.shape[1], 
                              returns_to_go.shape[-1]),
                              dtype=torch.float16,
                              device=returns_to_go.device), returns_to_go], dim=1)
            
            timesteps = torch.cat(
                [torch.zeros((timesteps.shape[0], 
                              self.seq_len-timesteps.shape[1], 
                              timesteps.shape[-1]), 
                             device=timesteps.device), timesteps], dim=1).to(torch.int32)
            
            detections = torch.cat(
                [torch.zeros((detections.shape[0], 
                              self.seq_len-detections.shape[1], 
                              detections.shape[-1]), dtype=torch.float16,
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
                                  target_waypoints.shape[-1]), dtype=torch.float16,
                                  device=target_waypoints.device), target_waypoints], dim=1)
            
        else:
            attention_mask = None
    
        # Note: prompt within kwargs
        action_preds, _, x = self.forward(
            img_states, actions, returns_to_go, timesteps, detections, decisions=decisions,
            target_waypoints=target_waypoints, attention_mask=attention_mask,
            prompt=prompt, q=q, train=train, task_id=task_id)
    
        return _, action_preds, _, x

def load_pretrained_model(model_path, load_8bit=False, load_4bit=False, device_map="auto", device="cuda"):

    from .mobilellama import MobileLlamaForCausalLM

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
