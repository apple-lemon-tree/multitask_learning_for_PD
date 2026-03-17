import numpy as np
import pandas as pd
import torch
from nltk import word_tokenize
from nltk.corpus import stopwords


class AffectiveFeatureExtractor:
    def __init__(self,
                 affective_space_path="../datasets/affectivespace/affectivespace.csv"):
        """
        初始化AffectiveSpace特征提取器
        :param affective_space_path: AffectiveSpace 100维嵌入文件路径（csv）
        """
        self.stop_words = set(stopwords.words('english'))

        # 加载 AffectiveSpace 100维嵌入
        self.affective_space = self._load_affective_space(affective_space_path)
        self.sentic_feature_dim = 100

    def _load_affective_space(self, file_path):
        """加载AffectiveSpace 100维嵌入为字典"""
        float_cols_dtype = {i: 'float32' for i in range(1, 101)}
        dtype_mapping = {
            0: 'str',
            **float_cols_dtype
        }
        df = pd.read_csv(
            filepath_or_buffer=file_path,
            header=None,
            encoding='utf-8',
            dtype=dtype_mapping,
            low_memory=False,
        )
        sent_dic = {row[0]: row[1:].values.astype(np.float32) for _, row in df.iterrows()}
        return sent_dic

    def extract_features_per_token(self, text):
        """
        提取文本中每个词的AffectiveSpace特征（逐词100维）
        若词不在词典中，则用全0向量代替。
        :param text: 输入文本字符串
        :return: numpy.ndarray，形状 [num_tokens, 100]
        """
        tokens = [
            token.lower() for token in word_tokenize(text)
            if token.isalpha() and token.lower() not in self.stop_words
        ]

        features = []
        zero_vec = np.zeros(self.sentic_feature_dim, dtype=np.float32)

        for token in tokens[:512]:
            vec = self.affective_space.get(token, zero_vec)
            features.append(vec)

        if not features:
            # 没有有效词时返回全0
            return np.zeros((1, self.sentic_feature_dim), dtype=np.float32)

        return np.stack(features, axis=0)  # [num_tokens, 100]

    def extract_features_batch(self, text,max_length):
        """
        将输入文本分词后，取前256个token，使用AffectiveSpace词典提取其100维特征，
        写入到形状为(256, 100)的零矩阵中（不足则其余保持为0）。
        :param text: str，单段文本
        :return: numpy.ndarray，形状为 (256, 100)
        """
        # 初始化 256 x 100 的全零矩阵
        feature_matrix = np.zeros((max_length, self.sentic_feature_dim), dtype=np.float32)

        # 分词与预处理（只保留字母token并去停用词）
        tokens = [
            token.lower() for token in word_tokenize(text)
            if token.isalpha() and token.lower() not in self.stop_words
        ]

        # 将前256个token的向量写入矩阵
        for index, token in enumerate(tokens[:max_length]):
            vec = self.affective_space.get(token)
            if vec is not None:
                feature_matrix[index] = np.asarray(vec, dtype=np.float32)
            # 若词不在词典，保持该行全0

        return feature_matrix
