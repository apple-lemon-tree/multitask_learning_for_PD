import torch

from torch import nn
from transformers import AutoModel
from mamba_ssm import Mamba

class SingletaskModel(nn.Module):
    def __init__(self,model_name):
        super().__init__()
        # model.config.use_cache = False
        # lora_config = LoraConfig(
        #     r=8,
        #     lora_alpha=32,
        #     lora_dropout=0.1,
        #     target_modules=[
        #         "q_proj",
        #         "k_proj",
        #         "v_proj",
        #         "o_proj",
        #     ],
        #     bias="none",
        #     task_type="SEQ_CLS"
        # )
        # self.bert = get_peft_model(model,lora_config)
        # self.bert.print_trainable_parameters()
        # self.bert = AutoModel.from_pretrained(model_name)
        self.electra = AutoModel.from_pretrained(model_name,trust_remote_code=True)

        self.dropout = nn.Dropout(0.3)
        self.personality_head = nn.Linear(768,5)

    def forward(self, input_ids, attention_mask):
        # embeddings = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        outputs = self.electra(input_ids=input_ids,attention_mask=attention_mask)
        cls = outputs.last_hidden_state[:,0,:]
        # masked_hidden = outputs * attention_mask.unsqueeze(0.1)
        # summed = masked_hidden.sum(dim=1)
        # lengths = attention_mask.sum(dim=1,keepdim=True).clamp(min=1)
        # result = summed / lengths
        # print(f"result.shape:{result.shape}")
        cls = self.dropout(cls)
        personality_logits = self.personality_head(cls)
        return personality_logits

