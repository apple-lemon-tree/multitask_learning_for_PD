import time

import numpy as np
import pandas as pd

import torch
from nltk import word_tokenize, ngrams
from nltk.corpus import stopwords
from nltk.sem.chat80 import concepts


class SingleTextAffectiveSpace:
    def __init__(self,
                 affective_space_path="affective.csv"):
        """
        初始化AffectiveSpace与BERT特征提取器
        :param affective_space_path: AffectiveSpace 100维嵌入文件路径
        :param senticnet_path: SenticNet8概念库文件路径
        :param bert_model_name: BERT模型名称
        """

        self.stop_words = set(stopwords.words('english'))
        self.max_ngram = 3  # 最大多词概念长度

        # 2. 加载AffectiveSpace 100维嵌入
        self.affective_space = self._load_affective_space(affective_space_path)

        self.sentic_feature_dim = 100  # 固定100维

    def _load_affective_space(self, file_path):
        """加载AffectiveSpace 100维嵌入"""
        # df = pd.read_excel(
        #     file_path,
        #     index_col=0,
        #     header=None,
        #     # dtype=np.float32,
        #     engine="openpyxl",
        #     usecols="A:CW",
        # )
        # return {concept: row for concept, row in df.iterrows()}

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
        sent_dic = {row[0]:row[1:] for _, row in df.iterrows()}
        return sent_dic

    def _extract_concepts(self, text, top_k=30):
        """从文本中提取匹配的SenticNet概念（支持多词概念）"""
        # 1. 文本预处理与分词
        tokens = [
            token.lower() for token in word_tokenize(text)
            if token.lower() not in self.stop_words and token.isalpha()
        ]
        N = len(tokens)
        # 2. 生成1-3词n-gram候选概念
        candidates = []
        for n in range(1, self.max_ngram + 1):
            for i in range(N - n + 1):
                span = tokens[i:i + n]
                concept = "_".join(span)
                if concept in self.affective_space:
                    candidates.append({
                        "concept": concept,
                        "start": i,
                        "end": i + n - 1,
                        "n": n
                    })

        if len(candidates) == 0:
            return []

        # 优先长概念排序
        candidates.sort(key=lambda x: x["n"], reverse=True)
        filtered = []

        for item in candidates:
            # 3. 通过 span 判断是否为子概念
            covered = False
            for f in filtered:
                if item["start"] >= f["start"] and item["end"] <= f["end"]:
                    covered = True
                    break

            if not covered:
                filtered.append(item)

            if len(filtered) >= top_k:
                break

        return [f["concept"] for f in filtered]

    def _extract_affective_features(self, concepts):
        """提取100维AffectiveSpace特征并聚合"""
        if not concepts:
            return np.zeros(self.sentic_feature_dim, dtype=np.float32)

        # 提取所有匹配概念的嵌入
        affective_embeddings = []
        for concept in concepts:
            affective_embeddings.append(self.affective_space[concept])

        affective_embeddings = np.array(affective_embeddings,dtype=np.float32)
        # 平均池化聚合为句子级特征
        return np.mean(affective_embeddings, axis=0).astype(np.float32)

    def extract_features_batch(self, text):
        """提取AffectiveSpace+BERT拼接特征"""

        # 1. 提取概念
        concepts = self._extract_concepts(text)
        # 2. 提取AffectiveSpace特征（100维）   应该是一个100维的向量
        affective_features = self._extract_affective_features(concepts)

        return affective_features

