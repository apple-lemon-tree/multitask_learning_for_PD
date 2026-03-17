from itertools import chain

import numpy as np
import torch
from transformers import (
    AutoModel, DistilBertModel, BertModel
)
import torch.nn as nn
from myutils.AffectiveSpaceFeatureExtractor import AffectiveFeatureExtractor
import torch.nn.functional as F

class MultitaskModel(nn.Module):
    def __init__(self,model_url,extractor,use_senticnet_feature=True,dropout=0.5):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_url)
        self.affective = extractor
        self.if_use_addition_feature = use_senticnet_feature

        input_dim = 868 if self.if_use_addition_feature else 768
        for param in self.bert.parameters():
            param.requires_grad = False

        for param in self.bert.encoder.layer[-6:].parameters():
            param.requires_grad = True

        n_filter = 256
        filter_sizes = [3,4,5]
        filter_sizes_PND = [3,4,5]
        output = 768
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=1,
                    out_channels=n_filter,
                    kernel_size=(fs, 768)
                )
                for fs in filter_sizes
            ]
        )

        self.dropout = nn.Dropout(0.2)
        self.sigmoid = nn.Sigmoid()
        self.isear_head = nn.Sequential(
            nn.Linear(input_dim,768),
            nn.Dropout(0.2),
            nn.Linear(768,7)
        )

        # nn.Linear(in_features=input_dim, out_features=7)
        self.personality_head = nn.Sequential(
            nn.Linear(input_dim,768),
            nn.Dropout(0.2),
            nn.Linear(768,5)
        )
            # nn.Linear(in_features=input_dim, out_features=5))



    def shared_parameters(self):

        bert_params = list(self.bert.parameters())
        convs_params = list(self.convs.parameters())
        return bert_params + convs_params


    def forward(self,affective_features1, input_ids1, attention_mask1,
                affective_features2=None, input_ids2=None, attention_mask2=None,
                is_eval=False):
        if is_eval == True:
            last_hidden_state1, pooled_output1 = self.bert(input_ids=input_ids1, attention_mask=attention_mask1,
                                                           return_dict=False)
            x1 = last_hidden_state1.unsqueeze(1)
            conved1 = [F.relu(conv(x1)).squeeze(3) for conv in self.convs]
            pooled1 = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved1]

            out1 = torch.cat(pooled1, dim=1)
            out1 = torch.cat((out1,affective_features1),dim=1)

            # out1 = self.dropout(out1)


            pnd_out = self.personality_head(out1)
            emo_out = self.isear_head(out1)
            return pnd_out, emo_out


        last_hidden_state1, pooled_output1 = self.bert(input_ids=input_ids1, attention_mask=attention_mask1, return_dict=False)
        # combined_features1 = torch.cat([affective_features1,outputs1],dim=1)

        last_hidden_state2, pooled_output2 = self.bert(input_ids=input_ids2, attention_mask=attention_mask2, return_dict=False)

        x1 = last_hidden_state1.unsqueeze(1) # (batch_size, 1, squence_length, embedding_dim)
        x2 = last_hidden_state2.unsqueeze(1) # (16, 1, 512, 768)

        conved1 = [F.relu(conv(x1)).squeeze(3) for conv in self.convs] # [(batch_size, n_filter, squence_length-fs+1),...]
        conved2 = [F.relu(conv(x2)).squeeze(3) for conv in self.convs] # [(16, 256, 510),(16,256,509),(16,256,508)]

        pooled1 = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved1] # [(batch_size,n_filter),...]
        pooled2 = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved2] # [(16,256),...]

        convedTo1 = [F.softmax(conv, dim=1)*conv for conv in pooled2] # [(16,256),...]
        convedTo2 = [F.softmax(conv, dim=1)*conv for conv in pooled1]


        pooled1 = [pooled1[i] + convedTo1[i] for i in range(len(pooled1))] # [(16,256),...]
        pooled2 = [pooled2[i] + convedTo2[i] for i in range(len(pooled2))]


        out1 = torch.cat(pooled1, dim=1) # (16,768+100)
        out1 = torch.cat((out1, affective_features1), dim=1)
        out2 = torch.cat(pooled2, dim=1)
        out2 = torch.cat((out2, affective_features2), dim=1)

        # out1 = self.dropout(out1)
        # out2 = self.dropout(out2)

        att1 = self.sigmoid(out1)
        att2 = self.sigmoid(out2)

        out1 = out1*att2
        out2 = out2*att1

        pnd_out = self.personality_head(out1)
        emo_out = self.isear_head(out2)

        return pnd_out,emo_out


