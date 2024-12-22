import torch
import numpy as np
import torch.nn as nn
import contextlib
from .mobilevlm import load_pretrained_model
from .mobilellama import MobileLlamaForCausalLM



class MobileVLM_Agent(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        
        self.config = config
        # init MobileVLM
        # model_path = config.model_path
        
        model_path = '/data2/pretrained-models/MobileVLM-1.7B'
        # TODO: use the official image_processor for alignment
        self.tokenizer, self.llm_model, self.image_processor, _ = load_pretrained_model(model_path, False, False)
        # load vit visual encoder
        self.visual_encoder = self.llm_model.get_vision_tower()
        # TODO: Discuss whether to freeze the parameters here
        self.vision_projector = self.llm_model.get_vision_projector()
    
    def maybe_autocast(self, dtype=torch.float16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")
        enable_autocast = True

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()
        
    def forward(self, states=None, actions=None, rtgs=None, timesteps=None,
                detections=None, decisions=None, target_waypoints=None,
                attention_mask=None, pen=False, train=False, task_id=None):
        
        self.device = device = torch.device('cuda')
        with self.maybe_autocast():

            # Here is the pseduo code:
            image_tensor = torch.randn(1, 3, 336, 336).to(device)
            
            # TODO:
            '''
            image_tensor = self.image_processor(image_tensor)
            image_tensor = self.image_processor.preprocess(image_tensor, return_tensors='np')['pixel_values'][0]
            image_np = np.ones((224, 224, 3)) * 255
            image_tensor = self.image_processor.preprocess(image_np, return_tensors='pt')['pixel_values'][0]
            '''
            vision_encoder = self.visual_encoder(image_tensor)
            vision_tokens = self.vision_projector(vision_encoder)
            
            llm_tokens = torch.randn(1, 30, 2048).to(device)        # [B, token_num, hidden_dim]
            llm_inputs = torch.cat((vision_tokens, llm_tokens), dim=1)
            llm_attention_mask = torch.ones((llm_inputs.shape[:2])).to(device)
            hidden_states = self.llm_model(
                    input_ids=None,
                    inputs_embeds=llm_inputs,
                    attention_mask=llm_attention_mask,
                    output_hidden_states=True,
                    return_dict=True,
                )
            return hidden_states

def attach_debugger():
    import debugpy
    debugpy.listen(5678)
    print("Waiting for debugger!")
    debugpy.wait_for_client()
    print("Attached!")
    

if __name__ == '__main__':
    attach_debugger()

    model = MobileVLM_Agent()
    output = model()