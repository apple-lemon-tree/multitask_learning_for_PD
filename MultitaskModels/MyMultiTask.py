import math
import torch
from transformers import (
    AutoModel, BertModel
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



# --- 2. 修改后的多任务模型 ---
class MultitaskModel(nn.Module):
    def __init__(self):
        super().__init__()
        n_filters = 256
        filter_sizes_EMO = [2, 3, 4]
        filter_sizes_PND = [2, 3, 4]

        # 注意：为了能进行Cross-stitch，两个任务的卷积核尺寸必须完全对应。
        # 这里 PND 和 EMO 都是 [2,3,4]，所以没问题。
        assert filter_sizes_EMO == filter_sizes_PND, "使用Cross-stitch时，两个任务的Kernel Sizes列表必须一致"

        output_dim = 768  # BERT output dim
        dropout = 0.3
        self.embedding_dim1 = 868  # 你代码中设定的维度
        self.embedding_dim2 = 768
        self.bert = AutoModel.from_pretrained('/hy-tmp/models/bert-base-uncased')
        # 1. 定义卷积层
        # 任务1 (PND)
        self.convs1 = nn.ModuleList([
            nn.Conv1d(in_channels=self.embedding_dim1, out_channels=n_filters, kernel_size=fs)
            for fs in filter_sizes_PND
        ])
        # 任务2 (EMO)
        self.convs2 = nn.ModuleList([
            nn.Conv1d(in_channels=self.embedding_dim2, out_channels=n_filters, kernel_size=fs)
            for fs in filter_sizes_EMO
        ])
        self.convs3 = nn.ModuleList([
            nn.Conv1d(in_channels=self.embedding_dim2, out_channels=n_filters, kernel_size=fs)
            for fs in filter_sizes_EMO
        ])

        # 2. 定义 Cross-stitch 单元
        # 我们需要两组 Stitch 单元，针对每一个 kernel size 都要有一对。

        self.stitch_shared1 = nn.ModuleList([
            CrossStitchUnit(num_tasks=2, init_weight=0.9)
            for _ in filter_sizes_PND
        ])
        self.stitch_shared2 = nn.ModuleList([
            CrossStitchUnit(num_tasks=2, init_weight=0.9)
            for _ in filter_sizes_PND
        ])

        self.dropout = nn.Dropout(dropout)

        # 输出层 (全连接层输入维度 = filter个数 * 卷积核种类数)
        fc_input_dim = n_filters * len(filter_sizes_PND)
        self.fcPDN = nn.Linear(fc_input_dim, 5)
        self.fcSent = nn.Linear(fc_input_dim, 7)
        self.fusion = CrossAttentionFusion()

        self.dimUP = nn.Linear(100, 768)

    def shared_parameters(self):
        return list(self.bert.parameters()) + list(self.convs3.parameters())
    def task_specific_parameters(self):
        a = [list(self.convs1.parameters()) +
             list(self.stitch_shared1.parameters()) +
             list(self.fcPDN.parameters()),
             list(self.convs2.parameters()) +
             list(self.stitch_shared2.parameters()) +
             list(self.fcSent.parameters())
             # list(self.fusion.parameters()) +
             # list(self.dimUP.parameters())
             ]


        return a


    def forward(self,input_ids1,attention_mask1,sentic_fea1,input_ids2, attention_mask2,sentic_fea2):
        # x1, x2 shape: [Batch, Seq_Len, Emb_Dim]
        x1 = self.bert(input_ids=input_ids1, attention_mask=attention_mask1)[0]
        x2 = self.bert(input_ids=input_ids2, attention_mask=attention_mask2)[0]

        # 拼接BERT特征和AffectiveSpace特征

        x1 = torch.cat((x1, sentic_fea1), dim=-1)
        sentic_fea2 = self.dimUP(sentic_fea2)
        x2 = self.fusion(x2,sentic_fea2)

        x3 = 0.5 * x1[:,:,:768] + 0.5 * x2
        # 维度转换: [Batch, Seq, Dim] -> [Batch, Dim, Seq] 以适配 Conv1d
        x1 = x1.permute(0, 2, 1)
        x2 = x2.permute(0, 2, 1)
        x3 = x3.permute(0, 2, 1)
        # 用于收集池化后的特征
        pooled_outputs_1 = []
        pooled_outputs_2 = []
        pooled_outputs_3 = []
        # --- 核心循环：逐个Kernel处理并应用Cross-stitch ---
        # 我们假设 convs1, convs2, convs_shared 的长度一致
        num_kernels = len(self.convs1)

        for i in range(num_kernels):
            # 1. 卷积 + 激活 (Feature Extraction)
            # Output shape: [Batch, n_filters, Length_new]
            feat_1 = F.relu(self.convs1[i](x1))
            feat_2 = F.relu(self.convs2[i](x2))
            feat_3 = F.relu(self.convs3[i](x3))
            # 2. 应用 Cross-stitch (Feature Fusion)
            # 这一步替代了原本的 shared_feature_fusion

            # [交互]: PND (x1) <-> EMO (x2)
            # feat_s 在这里会被更新，包含了 x1 的信息
            feat_1, feat_3 = self.stitch_shared1[i]([feat_1, feat_3])
            feat_2, feat_3 = self.stitch_shared2[i]([feat_2, feat_3])

            # 3. 池化 (Pooling)
            # Max-over-time pooling
            p1 = F.max_pool1d(feat_1, feat_1.shape[2]).squeeze(-1)
            p2 = F.max_pool1d(feat_2, feat_2.shape[2]).squeeze(-1)
            p3 = F.max_pool1d(feat_3, feat_3.shape[2]).squeeze(-1)
            pooled_outputs_1.append(p1)
            pooled_outputs_2.append(p2)
            pooled_outputs_3.append(p3)

        # --- 拼接特征 ---
        # cat shape: [Batch, n_filters * num_kernels]
        out1 = torch.cat(pooled_outputs_1, dim=1)
        out2 = torch.cat(pooled_outputs_2, dim=1)
        out3 = torch.cat(pooled_outputs_3, dim=1)


        out1 = out1 + out3
        out2 = out2 + out3
        # --- 融合 Shared 特征到任务特征 (可选) ---
        # 经过 Cross-stitch 后，out1 和 out2 其实已经包含了 shared 的信息。
        # 但通常我们还是会把 shared 的特征拼接到特定任务上，或者直接只用特定任务特征。
        # 这里的策略取决于你：
        # 策略 A: 只用 out1 和 out2 (因为它们已经“吸收”了 shared 的信息) -> 这是最标准的 Cross-stitch 用法
        # 策略 B: out1 = cat(out1, out_shared) -> 显式增强

        # 按照标准 Cross-stitch 网络，特征已经混合了，直接 dropout 即可：
        out1 = self.dropout(out1)
        out2 = self.dropout(out2)

        # 如果你想显式地再次利用 out_shared，可以解除下面这行的注释：
        # out1 = out1 + out_shared
        # out2 = out2 + out_shared

        # --- 分类 ---
        out_PDN = self.fcPDN(out1)
        out_sent = self.fcSent(out2)

        return out_PDN, out_sent
