from collections import Counter
import numpy as np
import pandas as pd


def print_emotion_label_distribution(labels, indices, subset_name):
    """
    计算并打印情感任务（7类）在给定索引下的标签分布。

    Args:
        labels (np.array): 所有的原始情感标签数组 (e_labels)。
        indices (list/np.array): 训练集或验证集的索引 (e_train_index/e_test_index)。
        subset_name (str): 子集名称（如 'Train' 或 'Validation'）。
    """
    subset_labels = labels[indices]

    # 情感任务 (isear) 是单标签，直接计数
    label_counts = Counter(subset_labels)
    total_samples = len(subset_labels)

    print(f"--- 情感任务 (ISEAR) {subset_name} 分布 (总数: {total_samples}) ---")

    # 对类别进行排序，使得输出更规整
    for label, count in sorted(label_counts.items()):
        percentage = (count / total_samples) * 100
        # 假设标签是从 0 开始的整数
        print(f"  类别 {label}: {count} 个 ({percentage:.2f}%)")

    print("-" * 50)
