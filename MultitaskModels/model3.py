import torch
from transformers import (
    AutoModel, BertModel
)
import torch.nn as nn
import torch.nn.functional as F

class TaskGating(nn.Module):
    def __init__(self, dim, dropout=0.0, use_layernorm=False):
        super().__init__()
        self.dim = dim
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.use_ln = use_layernorm

        self.Wa = nn.Linear(dim, dim)  # computes gate for A from hB
        self.Wb = nn.Linear(dim, dim)  # computes gate for B from hA

        if use_layernorm:
            self.lnA = nn.LayerNorm(dim)
            self.lnB = nn.LayerNorm(dim)
        else:
            self.lnA = None
            self.lnB = None

    def forward(self, hA, hB):

        # gate for A computed from hB; gate for B computed from hA
        gA = torch.sigmoid(self.Wa(hA))  # [N, D]
        gB = torch.sigmoid(self.Wb(hB))

        # gated fusion (residual-style)
        # can be hA' = hA + gA * hB  (you can also do gating* hB -> then combine differently)
        hA_out_flat = hA + gA * hB
        hB_out_flat = hB + gB * hA

        # optional dropout + layernorm
        hA_out_flat = self.dropout(hA_out_flat)
        hB_out_flat = self.dropout(hB_out_flat)

        if self.lnA is not None:
            hA_out_flat = self.lnA(hA_out_flat)
            hB_out_flat = self.lnB(hB_out_flat)

            hA_out = hA_out_flat
            hB_out = hB_out_flat

        return hA_out, hB_out

class CrossStitchUnit(nn.Module):
    def __init__(self, num_tasks=2, init_weight=0.9):
        super(CrossStitchUnit, self).__init__()
        self.num_tasks = num_tasks
        # 初始化权重：对角线为 0.9，其余为 0.1
        weights = torch.eye(num_tasks) * init_weight
        off_diag = (1. - init_weight) / (num_tasks - 1)
        weights[weights == 0] = off_diag
        self.weights = nn.Parameter(weights)

    def forward(self, inputs):
        # inputs: list of [tensor_task_a, tensor_task_b]
        # stack shape: (num_tasks, Batch, Channel, Length)
        stack = torch.stack(inputs, dim=0)
        # Einstein Summation: 自动处理后续维度
        outputs_stack = torch.einsum('nm, m...->n...', self.weights, stack)
        return torch.unbind(outputs_stack, dim=0)


class TextEmbedding(nn.Module):
    def __init__(self,model_url,use_SenticnetFeature):
        super(TextEmbedding, self).__init__()
        self.bert = AutoModel.from_pretrained(model_url, trust_remote_code=True)
        # self.sigmoid = nn.Sigmoid()
        

    def forward(self, input_ids, attention_mask, sentic_fea):
        embedding = self.bert(input_ids,attention_mask)
        lhs = embedding[0]
        # cls_emb = torch.max(lhs,dim=1)
        # weights = self.sigmoid(sentic_fea + lhs)
        # result = lhs + sentic_fea * weights

        result = torch.cat((lhs,sentic_fea),dim=2)
        # print("拼接后形状：",result.shape)
        return result


class MultitaskModel(nn.Module):
    def __init__(self):
        super().__init__()
        n_filters = 256
        # n_filters_PDN = 128
        filter_sizes_EMO = [2, 3, 4]
        filter_sizes_PND = [2, 3, 4]
        output_dim = 768
        dropout = 0.5
        self.embedding_dim = 868
        # 定义情感分析任务的卷积层
        self.convs1 = nn.ModuleList([
            nn.Conv1d(in_channels=self.embedding_dim,
                      out_channels=n_filters,
                      kernel_size=fs)
            for fs in filter_sizes_PND
        ])
        # 定义人格检测人物的卷积层
        self.convs2 = nn.ModuleList([
            nn.Conv1d(in_channels=self.embedding_dim,
                      out_channels=n_filters,
                      kernel_size=fs)
            for fs in filter_sizes_EMO
        ])

        self.convs_shared = nn.ModuleList([
            nn.Conv1d(in_channels=self.embedding_dim,
                      out_channels=n_filters,
                      kernel_size=fs)
            for fs in filter_sizes_PND
        ])

        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()

        # 输出层
        self.fcPDN = nn.Linear(768, 5)
        self.fcSent = nn.Linear(768, 7)

    def shared_feature_fusion(self,x1,x_share,x2):
        softmax_shared = [F.softmax(conv, dim=1) * conv for conv in x_share]
        softmax_v1 = [F.softmax(conv, dim=1) * conv for conv in x1]
        softmax_v2 = [F.softmax(conv, dim=1) * conv for conv in x2]
        task_fea1 = [x1[i] + softmax_shared[i] for i in range(3)]
        task_fea2 = [x2[i] + softmax_shared[i] for i in range(3)]
        x_shared = [x_share[i] + softmax_v1[i] + softmax_v2[i] for i in range(3)]

        return task_fea1, x_shared, task_fea2

    def forward(self, x1, x2):

        # 方式一
        # weights = self.sigmoid(x1 + x2)
        # x3 = weights * x1 + (1 - weights) * x2

        # 方式二
        # fusion = torch.cat((x1,x2),dim=2) # (8,256,868+868)--->(8,256,868)
        # x3 = self.task_feature_fusion(fusion)

        # 方式三
        x3 = x1 + x2


        x1 = x1.permute(0, 2, 1)
        x2 = x2.permute(0, 2, 1)
        x3 = x3.permute(0, 2, 1)


        # 提取特征
        conved1 = [F.relu(conv(x1)) for conv in self.convs1]
        conved_shared = [F.relu(conv(x3)) for conv in self.convs_shared]
        conved2 = [F.relu(conv(x2)) for conv in self.convs2]

        conved1, conved_shared, conved2 = self.shared_feature_fusion(conved1, conved_shared,conved2)
        # 计算任务间余弦相似度
        # cosine_value1 = [self.cosine_2(conved1[i], conved2[i]).mean(dim=1, keepdim=True) for i in range(len(conved1))]
        # print(cosine_value1[0].size())
        # self.cosine_value1 = torch.cat(cosine_value1, dim=1).mean(dim=1)
        # conved2 = [F.relu(conv(x2)).squeeze(3) for conv in self.convs2]
        # conv_n = [batch size, n_filters, sent len - filter_sizes[n]]

        # 池化层，降维
        pooled1 = [F.max_pool1d(conv, conv.shape[2]).squeeze(-1) for conv in conved1]
        # pooled_n = [batch size, n_filters]
        pooled_shared = [F.max_pool1d(conv, conv.shape[2]).squeeze(-1) for conv in conved1]
        pooled2 = [F.max_pool1d(conv, conv.shape[2]).squeeze(-1) for conv in conved2]

        # 任务间进一步交互
        pooled1, pooled_shared, pooled2 = self.shared_feature_fusion(pooled1, pooled_shared, pooled2)

        # 拼接特征并计算相似度
        out1 = torch.cat(pooled1, dim=1)
        out_shared = torch.cat(pooled_shared, dim=1)
        out2 = torch.cat(pooled2, dim=1)
        # self.cosine_value2 = self.cosine_1(out1, out2)

        out1 = self.dropout(out1)
        out_shared = self.dropout(out_shared)
        out2 = self.dropout(out2)

        # out1 = self.dropout(torch.cat(pooled1, dim=1))

        # 归一化
        # att1 = self.sigmoid(out1)
        # att2 = self.sigmoid(out2)
        # # att2 = F.softmax(out2, dim=1)
        # # att2_1 = torch.exp(out2)
        # # att2 = att2_1/ (1e-8 + (att2_1.sum(dim=1, keepdim=True)))
        #
        # # 任务间特征增强
        # out1 = out1 * att2  # *x1_att
        # out2 = out2 * att1  # *x2_att

        # 计算相似度
        # self.cosine_value3 = self.cosine_1(out1, out2)
        # out = torch.cat([out1, out2], dim=1)
        # out = out1 + out2
        out_PDN = self.fcPDN(out1)
        out_sent = self.fcSent(out2)


        return out_PDN, out_sent

