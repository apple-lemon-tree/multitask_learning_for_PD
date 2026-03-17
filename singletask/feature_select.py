import numpy as np
import pandas as pd
import torch
from MultitaskModels.model1 import PSOFeatureSelector
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_embeddings(data):
    tokenizer = AutoTokenizer.from_pretrained("/hy-tmp/models/bert-base-uncased")
    model = AutoModel.from_pretrained("/hy-tmp/models/bert-base-uncased").to(device)
    data = data.tolist()
    embeddings = []
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(data), 8),desc="embedding..."):
            inputs = tokenizer(data[i:min(i+8,len(data))],return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
            outputs = model(**inputs)
            emb = outputs.last_hidden_state[:,0,:].detach().cpu()
            embeddings.append(emb)
            del inputs, outputs
            torch.cuda.empty_cache()

    return torch.cat(embeddings,dim=0).numpy()

def get_data():
    data = pd.read_csv(
        "/hy-tmp/dataset/Essays/原始数据/essays_1.csv",
        header=0,
        encoding="utf-8",
        encoding_errors="ignore",
    )
    return data.iloc[:, 0].fillna('').astype(str).values, data.iloc[:,1:6].values

if __name__ == '__main__':
    # 1. 定义设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 2. 定义数据形状
    N = 2467  # 样本数
    D = 768  # 原始特征维度
    L = 5  # 标签数

    # 3. 目标降维维度
    K_TARGET = 128  # 假设我们想降维到 128 维

    # --- 4. 准备模拟数据 (请替换为您的真实数据) ---

    texts, labels = get_data()

    # X_data: (2000, 868)
    print("正在准备模拟数据...")
    X_data_full = get_embeddings(texts)
    # y_data: (2000, 5) - 多标签 (0 或 1)
    y_data_full = labels
    print(f"特征 X 形状: {X_data_full.shape}")
    print(f"标签 y 形状: {y_data_full.shape}")

    # 5. 初始化 PSO 选择器
    pso_selector = PSOFeatureSelector(
        n_particles=20,  # 粒子数 (太高会很慢)
        n_iterations=30,  # 迭代次数 (太高会很慢)
        target_dim=K_TARGET,
        X_data=X_data_full,
        y_data=y_data_full,
        device=device
    )

    # 6. 运行优化 (这将是最耗时的步骤)
    best_indices, best_score = pso_selector.optimize()

    # 7. 获取结果
    print("\n--- PSO 优化完成 ---")
    print(f"最佳特征子集索引 (前 20 个): {best_indices[:20]}...")
    print(f"最佳验证 F1 Score: {best_score:.4f}")

    # 8. 使用最佳索引提取降维后的特征
    X_reduced_final = X_data_full[:, best_indices]
    print(f"降维后的最终特征形状: {X_reduced_final.shape}")  # (2000, 128)