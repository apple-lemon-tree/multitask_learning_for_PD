import torch
from mamba_ssm import Mamba
from torch import nn
from transformers import AutoModel

class SingletaskModel(nn.Module):
    def __init__(self, embedding_matrix, freeze=False):
        super().__init__()
        num_layers = 24
        vocab_size, embedding_dim = embedding_matrix.shape
        self.embedding = nn.Embedding.from_pretrained(
            torch.FloatTensor(embedding_matrix),
            freeze=freeze,
            padding_idx=0
        )
        self.mamba = nn.ModuleList(
            Mamba(d_model=300, d_state=16, d_conv=4, expand=2)
            for _ in range(num_layers)
        )
        self.dropout = nn.Dropout(0.3)
        self.personality_head = nn.Linear(300, 5)

    def forward(self, input_ids,lengths):
        # input_ids: (batch_size, seq_len)
        embedded = self.embedding(input_ids) # (batch_size, MAX_SEQ_LEN, embedding_dim)

        for layer in self.mamba:
            embedded = layer(embedded) + embedded
        # ==========平均池化==========
        # mask = torch.arange(out.size(1), device="cuda")[None, :] < lengths[:, None]
        # mask = mask.unsqueeze(-1)
        # pooled = (out * mask).sum(dim=1) / lengths.unsqueeze(-1)
        # ===========================

        # =====最后一个有效token的隐状态=====
        idx = torch.arange(embedded.size(0), device=embedded.device)
        pooled = embedded[idx, lengths - 1]
        # ================================
        # x = self.dropout(torch.cat((cls,sentic_fea),dim=1))
        personality_logits = self.personality_head(pooled)
        return personality_logits

