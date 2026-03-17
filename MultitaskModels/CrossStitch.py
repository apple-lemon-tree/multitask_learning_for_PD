import torch
from transformers import (
    AutoModel, BertModel
)
import torch.nn as nn
import torch.nn.functional as F

class TextEmbedding(nn.Module):
    def __init__(self, model_url):
        super(TextEmbedding, self).__init__()
        self.bert = AutoModel.from_pretrained(model_url, trust_remote_code=True)

    def forward(self, input_ids, attention_mask, sentic_fea):
        embedding = self.bert(input_ids, attention_mask)
        lhs = embedding[0]

        result = torch.cat((lhs, sentic_fea), dim=2)

        return result

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
        self.embedding_dim = 868  # 你代码中设定的维度

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

        # 2. 定义 Cross-stitch 单元
        # 我们需要两组 Stitch 单元，针对每一个 kernel size 都要有一对。

        self.stitch_shared = nn.ModuleList([
            CrossStitchUnit(num_tasks=2, init_weight=0.9)
            for _ in filter_sizes_PND
        ])

        self.dropout = nn.Dropout(dropout)

        # 输出层 (全连接层输入维度 = filter个数 * 卷积核种类数)
        fc_input_dim = n_filters * len(filter_sizes_PND)
        self.fcPDN = nn.Linear(fc_input_dim, 5)
        self.fcSent = nn.Linear(fc_input_dim, 7)

    def forward(self, x1, x2):
        # x1, x2 shape: [Batch, Seq_Len, Emb_Dim]

        # 生成共享输入 (方式三)


        # 维度转换: [Batch, Seq, Dim] -> [Batch, Dim, Seq] 以适配 Conv1d
        x1 = x1.permute(0, 2, 1)
        x2 = x2.permute(0, 2, 1)

        # 用于收集池化后的特征
        pooled_outputs_1 = []
        pooled_outputs_2 = []

        # --- 核心循环：逐个Kernel处理并应用Cross-stitch ---
        # 我们假设 convs1, convs2, convs_shared 的长度一致
        num_kernels = len(self.convs1)

        for i in range(num_kernels):
            # 1. 卷积 + 激活 (Feature Extraction)
            # Output shape: [Batch, n_filters, Length_new]
            feat_1 = F.relu(self.convs1[i](x1))
            feat_2 = F.relu(self.convs2[i](x2))

            # 2. 应用 Cross-stitch (Feature Fusion)
            # 这一步替代了原本的 shared_feature_fusion

            # [交互]: PND (x1) <-> EMO (x2)
            # feat_s 在这里会被更新，包含了 x1 的信息
            feat_1, feat_2 = self.stitch_shared[i]([feat_1, feat_2])

            # 3. 池化 (Pooling)
            # Max-over-time pooling
            p1 = F.max_pool1d(feat_1, feat_1.shape[2]).squeeze(-1)
            p2 = F.max_pool1d(feat_2, feat_2.shape[2]).squeeze(-1)

            pooled_outputs_1.append(p1)
            pooled_outputs_2.append(p2)

        # --- 拼接特征 ---
        # cat shape: [Batch, n_filters * num_kernels]
        out1 = torch.cat(pooled_outputs_1, dim=1)
        out2 = torch.cat(pooled_outputs_2, dim=1)

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
