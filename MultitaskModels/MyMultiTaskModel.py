import math
import torch
from transformers import (
    AutoModel, BertModel, MAMConfig, PrefixTuningConfig, ParallelConfig
)
import torch.nn as nn
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

# --- 1. 定义 Cross-stitch 单元 ---
class CrossStitchUnit(nn.Module):
    def __init__(self, num_tasks=2, init_weight=0.9):
        super(CrossStitchUnit, self).__init__()
        # 初始化权重矩阵: 对角线为0.9 (保持自身), 非对角线为0.1 (交换)
        weights = torch.eye(num_tasks) * init_weight
        off_diag = (1. - init_weight) / (num_tasks - 1)
        weights[weights == 0] = off_diag
        self.weights = nn.Parameter(weights)

    def forward(self, inputs):
        # inputs: list of tensors [task_A, task_B]
        stack = torch.stack(inputs, dim=0)
        # 利用 einsum 进行加权组合
        outputs_stack = torch.einsum('nm, m...->n...', self.weights, stack)
        return torch.unbind(outputs_stack, dim=0)

class TextEmbedding(nn.Module):
    def __init__(self,model_url):
        super(TextEmbedding, self).__init__()
        self.bert = AutoModel.from_pretrained(model_url)
        # for param in self.bert.parameters():
        #     param.requires_grad = False
        #
        # adapters_config = ParallelConfig(
        #     reduction_factor=16,
        #     scaling=4.0,
        #     non_linearity="relu",
        #     is_parallel=True
        # )
        # prefix_tuning_config = PrefixTuningConfig(
        #     prefix_length=30,
        #     bottleneck_size=512,
        #     encoder_prefix=True,
        #     cross_prefix=False,
        #     flat=False
        # )
        #
        # mam_config = MAMConfig(
        #   adapter=adapters_config,
        #   prefix_tuning=prefix_tuning_config,
        # )
        # self.bert.add_adapter("MAM", config=mam_config)
        # self.bert.set_active_adapters("MAM")

    def forward(self,input_ids,attention_mask):

        outputs = self.bert(input_ids=input_ids,attention_mask=attention_mask)
        vec = outputs.last_hidden_state

        return vec


# --- 2. 修改后的多任务模型 ---
class MultitaskTextCNN(nn.Module):
    def __init__(self):
        super().__init__()
        n_filters = 256
        filter_sizes_EMO = [1, 2, 3]
        filter_sizes_PND = [1, 2, 3]

        # 注意：为了能进行Cross-stitch，两个任务的卷积核尺寸必须完全对应。
        # 这里 PND 和 EMO 都是 [2,3,4]，所以没问题。

        dropout = 0.3
        self.embedding_dim = 768
        self.output_dim = n_filters * len(filter_sizes_PND)
        # 1. 定义卷积层
        # 任务1 (PND)
        self.convs1 = nn.ModuleList([
            nn.Conv1d(in_channels=self.embedding_dim, out_channels=n_filters, kernel_size=fs)
            for fs in filter_sizes_PND
        ])
        # 任务2 (EMO)
        self.convs2 = nn.ModuleList([
            nn.Conv1d(in_channels=self.embedding_dim, out_channels=n_filters, kernel_size=fs)
            for fs in filter_sizes_EMO
        ])
        # self.convs3 = nn.ModuleList([
        #     nn.Conv1d(in_channels=self.embedding_dim, out_channels=n_filters, kernel_size=fs)
        #     for fs in filter_sizes_EMO
        # ])

        # 2. 定义 Cross-stitch 单元
        # 我们需要两组 Stitch 单元，针对每一个 kernel size 都要有一对。

        self.stitch_shared1 = nn.ModuleList([
            CrossStitchUnit(num_tasks=2, init_weight=0.9)
            for _ in filter_sizes_PND
        ])
        # self.stitch_shared2 = nn.ModuleList([
        #     CrossStitchUnit(num_tasks=2, init_weight=0.9)
        #     for _ in filter_sizes_PND
        # ])

        self.dense = nn.ModuleList([
            nn.Linear(self.output_dim, 768)
            for _ in range(2)
            ])

        self.dropout = nn.Dropout(dropout)

        self.fcPDN = nn.Linear(768, 5)
        self.fcSent = nn.Linear(868, 7)

        self.dimUP = nn.Linear(100, 768)

    def shared_parameters(self):
        return list(self.convs3.parameters())

    def task_specific_parameters(self):
        a = [list(self.convs1.parameters()) +
             list(self.stitch_shared1.parameters()) +
             list(self.dense[0].parameters()) +
             list(self.fcPDN.parameters()),

             list(self.convs2.parameters()) +
             list(self.stitch_shared2.parameters()) +
             list(self.dense[1].parameters()) +
             list(self.fcSent.parameters())
             ]

        return a

    def forward(self,x1,x2,sentic_feat1,sentic_feat2): # 人格输入，情绪输入
        # x1, x2 shape: [Batch, sentence_counts, 768+100]

        # x3 = torch.cat([x1,x2],dim=1)
        # 维度转换: [Batch, Seq, Dim] -> [Batch, Dim, Seq] 以适配 Conv1d
        x1 = x1.permute(0, 2, 1)
        x2 = x2.permute(0, 2, 1)
        # x3 = x3.permute(0, 2, 1)
        # 用于收集池化后的特征
        pooled_outputs_1 = []
        pooled_outputs_2 = []
        # pooled_outputs_3 = []
        # --- 核心循环：逐个Kernel处理并应用Cross-stitch ---
        # 我们假设 convs1, convs2, convs_shared 的长度一致
        num_kernels = len(self.convs1)

        for i in range(num_kernels):
            # 1. 卷积 + 激活 (Feature Extraction)
            # Output shape: [Batch, n_filters, Length_new]
            feat_1 = F.relu(self.convs1[i](x1))
            feat_2 = F.relu(self.convs2[i](x2))
            # feat_3 = F.relu(self.convs3[i](x3))
            # 2. 应用 Cross-stitch (Feature Fusion)
            # 这一步替代了原本的 shared_feature_fusion

            # 3. 池化 (Pooling)
            # Max-over-time pooling
            p1 = F.max_pool1d(feat_1, feat_1.shape[2]).squeeze(-1)
            p2 = F.max_pool1d(feat_2, feat_2.shape[2]).squeeze(-1)
            # p3 = F.max_pool1d(feat_3, feat_3.shape[2]).squeeze(-1)

            # [交互]: PND (x1) <-> EMO (x2)
            # feat_s 在这里会被更新，包含了 x1 的信息
            p1, p2 = self.stitch_shared1[i]([p1, p2])
            # p2, p3 = self.stitch_shared2[i]([p2, p3])

            pooled_outputs_1.append(p1)
            pooled_outputs_2.append(p2)
            # pooled_outputs_3.append(p3)

        # --- 拼接特征 ---
        # cat shape: [Batch, n_filters * num_kernels]
        out1 = torch.cat(pooled_outputs_1, dim=1)
        out2 = torch.cat(pooled_outputs_2, dim=1)
        # out3 = torch.cat(pooled_outputs_3, dim=1)


        out1 = self.dense[0](out1)
        out2 = self.dense[1](out2)
        # out3 = self.dense[2](out3)

        out1 = out1 + self.dimUP(sentic_feat1)
        out2 = torch.cat((out2, sentic_feat2), dim=1)

        final_out1 = self.dropout(out1)
        final_out2 = self.dropout(out2)
        # final_out3 = self.dropout(out3)

        # --- 分类 ---
        out_PDN = self.fcPDN(final_out1)

        # final_out2[:, :768] += final_out3
        out_sent = self.fcSent(final_out2)

        return out_PDN, out_sent
