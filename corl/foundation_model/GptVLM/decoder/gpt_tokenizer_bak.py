from transformers import GPT2Tokenizer, GPT2Model
import torch
import torch.nn as nn

# from ..LAVIS.lavis.models.blip2_models.Qformer import BertConfig,BertLMHeadModel
from .bert import BertLMHeadModel
from transformers import PretrainedConfig
# from transformers import AutoTokenizer, BertLMHeadModel

import contextlib

@torch.no_grad()
def get_text_embeddings(max_length=128):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model = GPT2Model.from_pretrained('gpt2')
    #TODO: modify the conversation here
    text = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. \
            You are a mature drive, you should drive along the street and turn right at the next T-junction, ful, detailed, and polite answers to the user's questions. \
            You are a mature drive, you should drive along the street and turn right at the next T-junction."
    
    encoded_input = tokenizer(text, return_tensors='pt', max_length=max_length, truncation=True, padding='longest')  # the length of the embedding is dynamic
    prompt_embeddings = model.get_input_embeddings()(encoded_input.input_ids)
    print(encoded_input.input_ids, encoded_input.attention_mask)
    print(prompt_embeddings.shape)
    return prompt_embeddings


class QFormer(nn.Module):
    vocab_size = 10000
    embed_size = 512
    num_heads = 8
    num_layers = 6
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(self.vocab_size, self.embed_size)
        self.transformer = nn.Transformer(
            d_model=self.embed_size,
            nhead=self.num_heads,
            num_encoder_layers=self.num_layers,
            num_decoder_layers=self.num_layers
        )
        self.output = nn.Linear(self.embed_size, self.vocab_size)
    
    def forward(self, query, context):
        query_embed = self.embedding(query)
        context_embed = self.embedding(context)
        transformer_output = self.transformer(query_embed, context_embed)
        output = self.output(transformer_output)
        return output
    

class BertConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`BertModel`] or a [`TFBertModel`]. It is used to
    instantiate a BERT model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the BERT
    [bert-base-uncased](https://huggingface.co/bert-base-uncased) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 30522):
            Vocabulary size of the BERT model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`BertModel`] or [`TFBertModel`].
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `Callable`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (`int`, *optional*, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        type_vocab_size (`int`, *optional*, defaults to 2):
            The vocabulary size of the `token_type_ids` passed when calling [`BertModel`] or [`TFBertModel`].
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        position_embedding_type (`str`, *optional*, defaults to `"absolute"`):
            Type of position embedding. Choose one of `"absolute"`, `"relative_key"`, `"relative_key_query"`. For
            positional embeddings use `"absolute"`. For more information on `"relative_key"`, please refer to
            [Self-Attention with Relative Position Representations (Shaw et al.)](https://arxiv.org/abs/1803.02155).
            For more information on `"relative_key_query"`, please refer to *Method 4* in [Improve Transformer Models
            with Better Relative Position Embeddings (Huang et al.)](https://arxiv.org/abs/2009.13658).
        is_decoder (`bool`, *optional*, defaults to `False`):
            Whether the model is used as a decoder or not. If `False`, the model is used as an encoder.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        classifier_dropout (`float`, *optional*):
            The dropout ratio for the classification head.

    Examples:

    ```python
    >>> from transformers import BertConfig, BertModel

    >>> # Initializing a BERT bert-base-uncased style configuration
    >>> configuration = BertConfig()

    >>> # Initializing a model (with random weights) from the bert-base-uncased style configuration
    >>> model = BertModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "bert"

    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        position_embedding_type="absolute",
        use_cache=True,
        classifier_dropout=None,
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.position_embedding_type = position_embedding_type
        self.use_cache = use_cache
        self.classifier_dropout = classifier_dropout

class VLMAdapter(nn.Module):
    def __init__(self, max_txt_len=64, num_token_queries=1, input_feat_dim=256, llm_hidden_dim=768):
        super().__init__()
        # load tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.max_txt_len = max_txt_len
        # init QFormer
        self.Qformer, self.query_tokens = self.init_Qformer(
                num_token_queries, input_feat_dim)
        self.Qformer.resize_token_embeddings(len(self.tokenizer))
        self.Qformer.cls = None
        self.llm_proj = nn.Linear(self.Qformer.config.hidden_size, llm_hidden_dim)
        self.llm_hidden_dim = llm_hidden_dim

        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.tokenizer.add_special_tokens({'bos_token': '</s>'})
        self.tokenizer.add_special_tokens({'eos_token': '</s>'})
        self.tokenizer.add_special_tokens({'unk_token': '</s>'})
    
    def maybe_autocast(self, dtype=torch.float16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        # enable_autocast = self.device != torch.device("cpu")
        enable_autocast = True

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()
    
    def init_Qformer(cls, num_query_token, vision_width, cross_attention_freq=2):
        encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        encoder_config.encoder_width = vision_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel.from_pretrained(
            "bert-base-uncased", config=encoder_config)
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return Qformer, query_tokens
    
    def transfer_vector_to_prompt(self, vector):
        # convert the provided instruction vector to a fixed prompt template
        B, t, c = vector.shape
        vector = vector.reshape(-1, c)
        prompt_list = []
        for i in range(B*t):
            instruction = vector[i]
            bike_flag = instruction[0]
            ped_flag = instruction[1]
            red_light_flag = instruction[2]
            stop_flag = instruction[3]
            target_waypoints = instruction[-2:].data.cpu().numpy()
            if bike_flag:
                bike_prompt = 'There are some bikes,'
            else:
                bike_prompt = ' There are no bikes,'
            
            if ped_flag:
                ped_prompt = ' some pedestrians,'
            else:
                ped_prompt = ' no pedestrians,'
            
            if red_light_flag:
                red_light_prompt = ' a red light,'
            else:
                red_light_prompt = ' no red light,'
            
            if stop_flag:
                stop_prompt = ' a stop sign in the scene'
            else:
                stop_prompt = ' no stop sign in the scene'
            
            target_pts_prompt = f', and the target waypoint is {target_waypoints[0]} {target_waypoints[1]}'
            current_prompt = bike_prompt + ped_prompt + red_light_prompt + stop_prompt + target_pts_prompt

            prompt_list.append(current_prompt)
        return prompt_list
    
    def concat_text_image_input(self, input_embeds, input_atts, image_embeds, image_atts=None):
        '''
        attention_mask:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        '''
        input_part_targets_len = []
        llm_inputs = []
        bs = image_embeds.size()[0]
        for i in range(bs):
            this_input_ones = input_atts[i].sum()
            input_part_targets_len.append(this_input_ones)
            if image_atts is None:
                bs, t, n, dim = image_embeds.size()
                llm_inputs.append(
                    torch.cat([
                        input_embeds[i][:this_input_ones],
                        image_embeds[i].view(t*n, -1),
                        input_embeds[i][this_input_ones:]
                    ])
                )
            else:
                llm_inputs.append(
                    torch.cat([
                        input_embeds[i][:this_input_ones],
                        image_embeds[i],
                        input_embeds[i][this_input_ones:]
                    ])
                )

        llm_inputs = torch.stack(llm_inputs, 0)
        return llm_inputs
    
    def forward(self, input_dict, get_embeddings_func):
        instruction_vector = input_dict['LanguageInstruction']        # [B, t, 6]
        input_embeds = input_dict['ImageTokenEmbedding']   # [B, t, n, c]
        B, t, n, c = input_embeds.shape
        input_embeds = input_embeds.view(B*t, n, c)
        device = input_embeds.device

        # convert the instruction vector to prompt
        prompt = self.transfer_vector_to_prompt(instruction_vector)
        query_tokens = self.query_tokens.expand(input_embeds.shape[0], -1, -1)
        text_Qformer = self.tokenizer(
            prompt,
            padding=True,
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(device)
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(device)
        Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask],dim=1)
        image_atts = torch.ones(input_embeds.size()[:-1], dtype=torch.long).to(device)
        
        query_output = self.Qformer.bert(
                text_Qformer.input_ids,
                attention_mask=Qformer_atts,
                query_embeds=query_tokens,
                encoder_hidden_states=input_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
        
        input_embeds = self.llm_proj(query_output.last_hidden_state[:, :query_tokens.size(1), :])
        input_embeds = input_embeds[:, None]

        # tokenizer the prompts and obtain the embeddings
        self.tokenizer.padding_side = "right"
        self.tokenizer.truncation_side = 'left'
        text_input_tokens = self.tokenizer(
            prompt,
            padding=True,
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            text_embeds = get_embeddings_func(text_input_tokens.input_ids)
        
        image_atts = None
        llm_inputs = self.concat_text_image_input(text_embeds, text_input_tokens.attention_mask, input_embeds, image_atts)

        # reshape the llm_input to original shape
        llm_inputs = llm_inputs.view(B, t, -1, self.llm_hidden_dim)
        
        return llm_inputs

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
    attach_debugger()

    # test QFormer
    test_input_data = torch.randn(1, 2, 20, 256).cuda()
    prompt = torch.randint(0, 2, (1, 2, 6)).cuda()

    input_dict = {
        'MDPEmbedding': test_input_data,
        'LanguageInstruction': prompt
    }
    # import pdb; pdb.set_trace()
    llm_model = GPT2Model.from_pretrained('gpt2', torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).cuda()

    adapter = VLMAdapter(llm_hidden_dim=768).cuda()
    hidden_state = adapter(input_dict, llm_model)



    