import math
import torch
from torch import nn
from transformers import AutoModel, MAMConfig,ParallelConfig,PrefixTuningConfig
import torch.nn.functional as F


class CrossAttentionFusion(nn.Module):
    def __init__(self, embed_dim=768, num_heads=8, dropout=0.3, output_proj=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        # Q from feat1, K/V from feat2
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(embed_dim, embed_dim) if output_proj else nn.Identity()

    def forward(self, feat1, feat2, key_padding_mask=None):
        """
        Args:
            feat1: [B, N, D]  --> Query
            feat2: [B, N, D]  --> Key & Value
            key_padding_mask: [B, N], True for padding positions (optional)
        Returns:
            fused: [B, N, D]
        """
        B, N, D = feat1.shape

        # Project to Q, K, V
        Q = self.q_proj(feat2)  # [B, N, D]
        K = self.k_proj(feat1)  # [B, N, D]
        V = self.v_proj(feat1)  # [B, N, D]

        # Reshape for multi-head: [B, N, D] -> [B, N, H, d] -> [B, H, N, d]
        Q = Q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, N, d]
        K = K.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, N, d]
        V = V.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, N, d]

        # Scaled dot-product attention
        scale = math.sqrt(self.head_dim)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / scale  # [B, H, N, N]

        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, N]
            attn_scores = attn_scores.masked_fill(mask, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, H, N, N]
        attn_weights = self.dropout(attn_weights)

        # Apply attention to V
        output = torch.matmul(attn_weights, V)  # [B, H, N, d]
        output = output.transpose(1, 2).contiguous().view(B, N, D)  # [B, N, D]

        output = self.out_proj(output)
        return output

class TextEmbedding(nn.Module):
    def __init__(self,model_url):
        super(TextEmbedding, self).__init__()
        # adapters_config = ParallelConfig(
        #     reduction_factor=16,
        #     scaling=4.0,
        #     non_linearity="relu",
        #     is_parallel=True
        # )
        # prefix_tuning_config = PrefixTuningConfig(
        #     prefix_length=30,
        #     bottleneck_size=800,
        #     encoder_prefix=True,
        #     cross_prefix=False,
        #     flat=False
        # )
        self.bert = AutoModel.from_pretrained(model_url)


    def forward(self,input_ids,attention_mask):

        outputs = self.bert(input_ids=input_ids,attention_mask=attention_mask)
        vec = outputs.last_hidden_state

        return vec

class SingleTaskTextCNN(nn.Module):
    def __init__(self):
        super().__init__()
        n_filters = 256
        filter_sizes= [2, 3, 4]
        dropout = 0.3
        self.embedding_dim = 768
        # 定义情感分析任务的卷积层
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=self.embedding_dim,
                      out_channels=n_filters,
                      kernel_size=fs)
            for fs in filter_sizes
        ])
        self.dense = nn.Linear(768,768)
        self.dropout = nn.Dropout(dropout)
        # 输出层
        self.fcPDN = nn.Linear(768, 5)
        # self.dimup = nn.Linear(100,768)


    def forward(self, x):

        x = x.permute(0, 2, 1)
        # 提取特征
        conved = [F.relu(conv(x)) for conv in self.convs]

        # 池化层，降维
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(-1) for conv in conved]

        out = torch.cat(pooled, dim=1)

        out = self.dense(out)

        # out = torch.cat([out,torch.tanh(sentic_feat)], dim=-1)

        final_out = self.dropout(out)

        out_PDN = self.fcPDN(final_out)
        return out_PDN
