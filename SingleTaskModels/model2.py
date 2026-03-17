import torch
from torch import nn
from transformers import AutoModel,BertForSequenceClassification


class SingletaskModel(nn.Module):
    def __init__(self,url):
        super().__init__()
        self.bert = AutoModel.from_pretrained(url)

        self.dropout = nn.Dropout(0.3)
        self.emotion_head = nn.Linear(768, 1)

    def forward(self, input_ids, attention_mask):

        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]

        x = self.dropout(pooled_output)
        emotion_logits = self.emotion_head(x)
        return emotion_logits

