import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer, BertModel

drop_out = 0.3


class AttentionPooling(nn.Module):
    def __init__(self, hidden_size=768):
        super().__init__()
        self.W_att = nn.Linear(hidden_size, 1)

    def forward(self, embeddings):
        """
        参数:
        embeddings (torch.Tensor): 形状 [B, N, H]
            (B: Batch Size, N: Num Chunks, H: Hidden Size)
        """

        # 1. 计算未归一化的注意力分数 (Energy)
        # energy 形状: [B, N, 1]
        energy = self.W_att(embeddings)

        # 2. 移除最后一维 (1) 并应用 Softmax
        # attention_weights 形状: [B, N]
        attention_weights = torch.softmax(energy.squeeze(-1), dim=1)

        # 3. 将权重形状调整回 [B, N, 1]，以便进行按元素乘法 (Element-wise multiplication)
        # attention_weights 形状: [B, N, 1]
        attention_weights = attention_weights.unsqueeze(-1)

        # 4. 计算加权求和
        # weighted_embeddings 形状: [B, N, H]
        weighted_embeddings = embeddings * attention_weights

        # 5. 沿着分块维度 (dim=1) 求和
        # document_embedding 形状: [B, H]
        document_embedding = torch.sum(weighted_embeddings, dim=1)

        return document_embedding

class TextEmbedding(nn.Module):
    def __init__(self,model_url):
        super(TextEmbedding, self).__init__()
        self.bert = AutoModel.from_pretrained(model_url, trust_remote_code=True)
        self.sentic_layer = nn.Sequential(
            nn.LayerNorm(100),
            nn.Linear(100,384),
            nn.GELU(),
            nn.Linear(384,512),
            nn.GELU(),
            nn.Linear(512,768),
        )

    def forward(self, input_ids, attention_mask, sentic_fea):
        embedding = self.bert(input_ids,attention_mask)
        sentic_vec = self.sentic_layer(sentic_fea)
        cls_emb = embedding.last_hidden_state[:, 0, :]
        return cls_emb + sentic_vec# 数值上相加
# class TextEmbedding(nn.Module):
#     def __init__(self,model_url):
#         super(TextEmbedding, self).__init__()
#         self.bert = AutoModel.from_pretrained(model_url, trust_remote_code=True)
#         self.bert_hidden_size = self.bert.config.hidden_size
#         self.attention_pooling = AttentionPooling(self.bert_hidden_size)
#
#     def forward(self, input_ids, attention_mask, sentic_fea):
#         # 获取batch_size, num_chunks, seq_length
#         if input_ids.dim() == 3:
#             B, N, L = input_ids.shape
#
#             # 扁平化以适应bert输入
#             input_ids_flat = input_ids.view(B * N, L)
#             attention_mask_flat = attention_mask.view(B * N, L)
#
#             # bert编码
#             outputs = self.bert(input_ids=input_ids_flat,attention_mask=attention_mask_flat)
#             cls_embeddings_flat = outputs.last_hidden_state[:, 0, :]
#
#             # 聚合 cls_embedding_unflat (B,N,H)
#             cls_embedding_unflat = cls_embeddings_flat.view(B, N, self.bert_hidden_size)
#
#             # 池化
#             document_bert_embeddings = self.attention_pooling(cls_embedding_unflat)
#
#             # 拼接
#             final_embeddings = torch.cat((document_bert_embeddings, sentic_fea), dim=1)
#             return final_embeddings
#         elif input_ids.dim() == 2:
#             outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
#             cls = outputs.last_hidden_state[:, 0, :]
#             final_embeddings = torch.cat((cls, sentic_fea), dim=1)
#             return final_embeddings



class Expert(nn.Module):
    def __init__(self, input_dim, expert_dim):
        super(Expert, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_dim, expert_dim),
            nn.LayerNorm(expert_dim),
            nn.GELU(),
            nn.Dropout(drop_out)
        )

    def forward(self, x):
        return self.layer(x)


class Gate(nn.Module):
    def __init__(self, input_dim, n_experts):
        super(Gate, self).__init__()
        self.gate = nn.Sequential(
            nn.Linear(input_dim, n_experts),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        weights = self.gate(x)  # [B, n_experts]
        return weights.unsqueeze(-1)  # [B, n_experts, 1]

class SoGgate(nn.Module):
    def __init__(self, expert_output):
        super(SoGgate, self).__init__()
        self.sog = nn.Sequential(
            nn.Linear(expert_output, expert_output),
            nn.Softmax(dim=-1)
        )
    def forward(self, x):
        weights = self.sog(x)
        return weights

class Tower(nn.Module):
    def __init__(self, expert_dim, num_labels,dropout):
        super(Tower, self).__init__()
        self.head = nn.Sequential(
            nn.Linear(expert_dim, expert_dim),
            nn.LayerNorm(expert_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(expert_dim, num_labels)
        )
        # self.norm = nn.LayerNorm(expert_dim)
        # self.head = nn.Linear(expert_dim, num_labels)

    def forward(self, x):
        # out = self.tower(x)
        # x = x + out
        # x = self.norm(x)
        out = self.head(x)
        return out

class PLELayer(nn.Module):
    def __init__(self, input_dim, expert_dim, n_tasks, n_task_experts, n_shared_experts):
        super(PLELayer, self).__init__()
        self.n_tasks = n_tasks
        self.task_experts = nn.ModuleList([
            nn.ModuleList([Expert(input_dim, expert_dim) for _ in range(n_task_experts)])
            for _ in range(n_tasks)
        ])
        self.shared_experts = nn.ModuleList([
            Expert(input_dim, expert_dim) for _ in range(n_shared_experts)
        ])
        self.task_gates = nn.ModuleList([
            SoGgate(expert_dim)
            for _ in range(n_tasks)
        ])
        self.shared_gate = SoGgate(expert_dim)

        self.task_norms = nn.ModuleList([
            nn.LayerNorm(expert_dim) for _ in range(n_tasks)
        ])

        self.shared_norm = nn.LayerNorm(expert_dim)

    def forward(self, task_inputs, shared_input):
        # Compute expert outputs

        task_outputs = []
        for i in range(self.n_tasks):
            task_outputs.append([expert(task_inputs[i]) for expert in self.task_experts[i]])
        shared_outputs = [expert(shared_input) for expert in self.shared_experts]
        # task_outputs [[专家1输出,专家2输出], [专家1输出,专家2输出]]    shared_out [专家1输出,专家2输出]
        # print(f"任务输出形状：{task_outputs[0][0].shape},共享单元形状：{shared_outputs[0].shape}")  (8,256) (8,256)
        # Task-specific gate outputs
        next_task_inputs = []
        for i in range(self.n_tasks):
            total = []
            for j in range(2):
                if j == 0:
                    shared = torch.stack(shared_outputs) # shared_outputs [tensor A,tensor B]
                    weights = self.task_gates[i](shared)
                    m = weights * shared # m (2,8,256)
                    # 两个东西是张量，所以‘+’起的作用是逐元素相加
                    h = torch.stack(task_outputs[i]) + m    # (2,8,256)
                    total.append(h)
                elif j == 1:
                    task_out = torch.stack(task_outputs[i])
                    weights = self.task_gates[i](task_out)
                    m = weights * task_out
                    # 两个东西是张量，所以‘+’起的作用是逐元素相加
                    h = torch.stack(shared_outputs) + m
                    total.append(h)
            # total [(2,8,768),(2,8,768)]  2*2*8*256
            stack1 = torch.stack(total) # (2,2,8,256)
            result = stack1.sum(dim=(0,1)) # result (8,256)
            next_task_inputs.append(result)

        # Shared gate output (for next layer's shared input)
        flat_all_tasks = torch.stack([torch.stack(task_outputs[0]).sum(0),torch.stack(task_outputs[1]).sum(0)])
        shared_weights = self.shared_gate(flat_all_tasks)
        m = shared_weights * flat_all_tasks
        next_shared_input = torch.stack(shared_outputs) + m  # [2,8,256]
        next_shared_input = next_shared_input.sum(dim=0)
        return next_task_inputs, next_shared_input


class PLE(nn.Module):
    # 正确处理多层维度
    def __init__(self, input_dim, expert_dim, n_tasks=2, n_layers=2,
                 n_task_experts=2, n_shared_experts=2):
        super(PLE, self).__init__()
        self.n_tasks = n_tasks
        self.ple_layers = nn.ModuleList()

        # 为每一层设置正确的输入维度
        for layer_idx in range(n_layers):
            if layer_idx == 0:
                # 第一层：使用原始输入维度
                current_input_dim = input_dim # 768
            else:
                # 后续层：使用expert输出维度作为输入
                current_input_dim = expert_dim # 768

            self.ple_layers.append(
                PLELayer(
                    input_dim=current_input_dim,  # 动态设置输入维度
                    expert_dim=expert_dim,
                    n_tasks=n_tasks,
                    n_task_experts=n_task_experts,
                    n_shared_experts=n_shared_experts
                )
            )

        self.tower1 = Tower(expert_dim, 5,0.5)
        self.tower2 = Tower(expert_dim, 7,0.3)

    def shared_params(self):
        params = []
        for ple_layer in self.ple_layers:
            params += list(ple_layer.parameters())
        return params

    def task_specific_params(self):
        return list(self.tower1.parameters()) + list(self.tower2.parameters())

    def forward(self, x1, x2):

        # Initial input: shared across all tasks and shared experts
        # task_inputs = [x for _ in range(self.n_tasks)]
        task_inputs = [x1, x2]
        shared_input = 0.5 * x1 + 0.5 * x2

        for layer in self.ple_layers:
            task_inputs, shared_input = layer(task_inputs, shared_input)

        out1 = self.tower1(task_inputs[0])
        out2 = self.tower2(task_inputs[1])

        # return task_inputs  # final task-specific vectors [task1_repr, task2_repr, task3_repr]
        return out1, out2