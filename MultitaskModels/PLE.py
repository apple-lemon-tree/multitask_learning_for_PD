import numpy as np
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer, BertModel

drop_out = 0.5

class TextEmbedding(nn.Module):
    def __init__(self,model_url):
        super(TextEmbedding, self).__init__()
        self.bert = AutoModel.from_pretrained(model_url)

    def forward(self, input_ids, attention_mask):
        embedding = self.bert(input_ids,attention_mask)
        cls_emb = embedding.last_hidden_state[:, 0, :]

        return cls_emb

class Expert(nn.Module):
    def __init__(self, input_dim, expert_dim):# 768 768
        super(Expert, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_dim, expert_dim),
            nn.ReLU(),
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

class Tower(nn.Module):
    def __init__(self, expert_dim, num_labels,dropout):
        super(Tower, self).__init__()
        self.head = nn.Linear(expert_dim, num_labels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(x)
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
            Gate(input_dim, n_task_experts + n_shared_experts)
            for _ in range(n_tasks)
        ])
        self.shared_gate = Gate(input_dim, n_tasks * n_task_experts + n_shared_experts)

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

        # Task-specific gate outputs
        next_task_inputs = []
        for i in range(self.n_tasks):
            all_expert_outputs = task_outputs[i] + shared_outputs # task_outputs[i] [[16*768], [16*768]]每个任务两个专家输出
            stacked = torch.stack(all_expert_outputs, dim=1)  # [batch_size, experts, hidden_size] (16,任务i的专家数量+共享的专家数量,768)
            weights = self.task_gates[i](task_inputs[i])  # [B, n_experts, 1] (16,任务i的专家数量+共享的专家数量,1)

            fused = torch.sum(stacked * weights, dim=1)  # [B, D]
            fused = self.task_norms[i](fused)
            next_task_inputs.append(fused)

        # Shared gate output (for next layer's shared input)
        flat_all_experts = sum(task_outputs, []) + shared_outputs
        stacked_shared = torch.stack(flat_all_experts, dim=1)
        shared_weights = self.shared_gate(shared_input)
        next_shared_input = torch.sum(stacked_shared * shared_weights, dim=1)  # [B, D]
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
                current_input_dim = input_dim # 868
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

        self.tower1 = Tower(expert_dim, 5,0.3)
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