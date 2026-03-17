import torch
from torch import nn
from transformers import AutoModel

import numpy as np
import time
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, accuracy_score

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# 定义一个简单的多标签分类头，用于评估
class SimpleMultilabelHead(nn.Module):
    def __init__(self, input_dim, num_labels):
        super().__init__()
        # input_dim 将是 K (例如 128)
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_labels)
        )
    def forward(self, x):
        return self.classifier(x)

class PSOFeatureSelector:
    def __init__(self, n_particles, n_iterations, target_dim,
                 X_data, y_data, device):
        """
        初始化PSO
        :param n_particles: 粒子数量 (例如 20)
        :param n_iterations: 迭代次数 (例如 50)
        :param target_dim: 目标降维后的维度 K (例如 128)
        :param X_data: 完整特征 (N=2000, D=868), 必须是 NumPy 数组
        :param y_data: 完整标签 (N=2000, L=5), 必须是 NumPy 数组
        :param device: PyTorch 设备 ('cuda' 或 'cpu')
        """
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.D = X_data.shape[1]  # 原始维度 868
        self.K = target_dim  # 目标维度 K
        self.X = X_data
        self.y = y_data
        self.device = device

        # PSO 核心参数
        self.w = 0.729  # 惯性权重
        self.c1 = 1.49  # 认知系数
        self.c2 = 1.49  # 社会系数

        # 初始化粒子 (在 CPU 上)
        # 位置：(n_particles, K) - 存储 K 个特征索引
        self.position = self._initialize_positions()
        # 速度：(n_particles, K)
        self.velocity = np.random.rand(n_particles, self.K) * 0.1

        # pbest 和 gbest (在 CPU 上)
        self.pbest_position = np.copy(self.position)
        self.pbest_fitness = np.full(n_particles, -np.inf)

        self.gbest_position = None
        self.gbest_fitness = -np.inf

    def _initialize_positions(self):
        """初始化 K 个随机且唯一的特征索引"""
        positions = np.zeros((self.n_particles, self.K), dtype=int)
        for i in range(self.n_particles):
            positions[i] = np.random.choice(self.D, self.K, replace=False)
            positions[i].sort()  # 排序以便于处理
        return positions

    def _evaluate_fitness(self, position):
        """
        适应度函数 (在 GPU 上运行)
        :param position: K 个特征索引 [idx1, idx2, ...]
        :return: Macro F1 Score (越高越好)
        """

        num_labels = self.y.shape[1]  # L=5

        # 1. CPU 上提取特征子集
        X_subset = self.X[:, position]

        # 2. 转换为 PyTorch 张量
        X_tensor = torch.tensor(X_subset, dtype=torch.float32)
        y_tensor = torch.tensor(self.y, dtype=torch.float32)

        # 3. K-Fold 交叉验证
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        f1_scores = []

        for train_index, val_index in kf.split(X_tensor):
            # 4. 初始化评估器 (每次都必须重新初始化)
            evaluator = SimpleMultilabelHead(self.K, num_labels).to(self.device)
            optimizer = torch.optim.Adam(evaluator.parameters(), lr=1e-4)
            criterion = nn.BCEWithLogitsLoss()  # 适用于多标签

            # 5. 转移数据到 GPU
            X_train, X_val = X_tensor[train_index].to(self.device), X_tensor[val_index].to(self.device)
            y_train, y_val = y_tensor[train_index].to(self.device), y_tensor[val_index].to(self.device)

            train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=16, shuffle=True)

            # --- 6. 快速训练 (GPU 加速) ---
            evaluator.train()
            # 注意：Epoch 数不宜过多，否则 PSO 会非常慢
            for epoch in range(5):
                for X_batch, y_batch in train_loader:
                    optimizer.zero_grad()
                    logits = evaluator(X_batch)
                    loss = criterion(logits, y_batch)
                    loss.backward()
                    optimizer.step()

            # --- 7. 快速评估 (GPU 加速) ---
            evaluator.eval()
            with torch.no_grad():
                val_logits = evaluator(X_val)
                # 预测 (Sigmoid > 0.5)
                val_preds = (torch.sigmoid(val_logits) > 0.5).cpu().numpy()
                val_true = y_val.cpu().numpy()

                # score = f1_score(val_true, val_preds, average='macro', zero_division=0)
                score = accuracy_score(val_true.flatten(), val_preds.flatten())
                f1_scores.append(score)

        # 8. 返回平均 F1 Score
        return np.mean(f1_scores)

    def optimize(self):
        """运行 PSO 优化 (在 CPU 上)"""

        print("--- 开始 PSO 特征选择 (使用 GPU 加速适应度评估) ---")

        for iteration in range(self.n_iterations):
            start_iter_time = time.time()

            for i in range(self.n_particles):
                # 1. 评估适应度 (昂贵步骤，使用 GPU)
                current_fitness = self._evaluate_fitness(self.position[i])

                # 2. 更新 pbest
                if current_fitness > self.pbest_fitness[i]:
                    self.pbest_fitness[i] = current_fitness
                    self.pbest_position[i] = self.position[i].copy()

                # 3. 更新 gbest
                if current_fitness > self.gbest_fitness:
                    self.gbest_fitness = current_fitness
                    self.gbest_position = self.position[i].copy()

            # 4. 更新所有粒子的速度和位置 (在 CPU 上)
            for i in range(self.n_particles):
                r1 = np.random.rand(self.K)
                r2 = np.random.rand(self.K)

                # 速度更新
                cognitive_v = self.c1 * r1 * (self.pbest_position[i] - self.position[i])
                social_v = self.c2 * r2 * (self.gbest_position - self.position[i])
                self.velocity[i] = self.w * self.velocity[i] + cognitive_v + social_v

                # 位置更新 (连续 -> 离散)
                self.position[i] = (self.position[i] + self.velocity[i]).astype(int)

                # 5. 边界和唯一性处理 (在 CPU 上)
                # 确保索引在 [0, D-1] 范围内
                self.position[i] = np.clip(self.position[i], 0, self.D - 1)

                # 确保 K 个索引是唯一的
                unique_indices = np.unique(self.position[i])
                if len(unique_indices) < self.K:
                    # 如果碰撞导致索引少于 K 个，则随机补充
                    missing_count = self.K - len(unique_indices)
                    available = np.setdiff1d(np.arange(self.D), unique_indices)
                    if len(available) < missing_count:
                        # 极端情况：可用索引不足，随机重复选择
                        new_indices = np.random.choice(available, missing_count, replace=True)
                    else:
                        new_indices = np.random.choice(available, missing_count, replace=False)
                    self.position[i] = np.concatenate([unique_indices, new_indices])

                # 确保最终是 K 个并排序
                self.position[i] = np.sort(self.position[i][:self.K])

            iter_time = time.time() - start_iter_time
            print(f"迭代 {iteration + 1}/{self.n_iterations} | "
                  f"最佳 F1 Score: {self.gbest_fitness:.4f} | "
                  f"耗时: {iter_time:.2f} 秒")

        return self.gbest_position, self.gbest_fitness
