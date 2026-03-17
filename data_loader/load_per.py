import numpy
import numpy as np
import pandas as pd

import torch
from nltk import word_tokenize, ngrams
from nltk.corpus import stopwords

from torch.utils.data import Dataset,DataLoader
from transformers import BertTokenizer, BertModel

short_length = 128
long_length= 512

df = pd.read_excel('D:/PythonProject/senticnet/senticnet.xlsx')
unique_labels = df['PRIMARY EMOTION'].unique()

sorted_labels = sorted(unique_labels)

emotion_id = {label: idx for idx, label in enumerate(sorted_labels)}
class AffectiveBERTFeatureExtractor:
    def __init__(self,
                 affective_space_path="D:/datasets_1/affectivespace/affectivespace.xlsx",
                 senticnet_path="D:/PythonProject/senticnet/senticnet8.py",
                 bert_model_name="D:/models/bert-base-uncased"):
        """
        初始化AffectiveSpace与BERT特征提取器
        :param affective_space_path: AffectiveSpace 100维嵌入文件路径
        :param senticnet_path: SenticNet8概念库文件路径
        :param bert_model_name: BERT模型名称
        """
        # 1. 加载SenticNet8概念库
        self.senticnet = self._load_senticnet(senticnet_path)
        self.stop_words = set(stopwords.words('english'))
        self.max_ngram = 3  # 最大多词概念长度

        # 2. 加载AffectiveSpace 100维嵌入
        self.affective_space = self._load_affective_space(affective_space_path)
        self.sentic_feature_dim = 100  # 固定100维
        # 3. 初始化BERT模型与分词器
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.bert_model = BertModel.from_pretrained(
            bert_model_name,
            output_hidden_states=False
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bert_model.to(self.device)

    def _load_senticnet(self, file_path):
        """加载SenticNet8概念库"""
        import sys
        from pathlib import Path
        # 将文件所在目录添加到Python路径
        sys.path.append(str(Path(file_path).parent))
        module_name = Path(file_path).stem  # 文件名（不含.py）
        senticnet_module = __import__(module_name)
        return senticnet_module.senticnet

    def _load_affective_space(self, file_path):
        """加载AffectiveSpace 100维嵌入"""
        # 读取CSV文件（无表头，第一列为概念，后100列为维度值）
        # df = pd.read_csv(
        #     file_path,
        #     header=None,
        #     encoding="latin1",
        #     encoding_errors="ignore",
        # )

        df = pd.read_excel(
            file_path,
            index_col=0,
            header=None,
            # dtype=np.float32,
            engine="openpyxl",
            usecols="A:CW",
        )
        # 转换为字典 {概念: 100维向量}
        return {concept: row.values for concept, row in df.iterrows()}

        # 转换为字典 {概念: 100维向量}
        #return {row[0]: row[1:101] for _, row in df.iterrows()}

    def _extract_concepts(self, text):
        """从文本中提取匹配的SenticNet概念（支持多词概念）"""
        # 1. 文本预处理与分词
        tokens = [
            token.lower() for token in word_tokenize(text)
            if token.lower() not in self.stop_words and token.isalpha()
        ]

        # 2. 生成1-3词n-gram候选概念
        candidates = []
        for n in range(1, self.max_ngram + 1):
            candidates.extend(ngrams(tokens, n))

        # 3. 匹配SenticNet和AffectiveSpace中的概念
        matched_concepts = []
        for candidate_tuple in candidates:
            concept = "_".join(candidate_tuple)
            # 确保概念同时存在于两个库中
            if concept in self.senticnet and concept in self.affective_space:
                is_sub_concept = False
                for existing_concept in matched_concepts:
                    if concept in existing_concept:
                        is_sub_concept = True
                        break
                if not is_sub_concept:
                    matched_concepts.append(concept)


        # 4. 去重并按长度排序（优先保留长概念）
        matched_concepts.sort(key=lambda x: len(x.split("_")), reverse=True)

        return matched_concepts[:10]  # 最多保留10个核心概念

    def _extract_affective_senticnet_features(self, concepts):
        """提取100维AffectiveSpace特征并聚合"""
        if not concepts:
            return np.zeros(self.sentic_feature_dim, dtype=np.float32)

        # 提取所有匹配概念的嵌入
        affective_embeddings = []
        senticnet_embeddings = []
        for concept in concepts:
            affective_embeddings.append(self.affective_space[concept])

            # info = []
            # data = self.senticnet[concept]

            # info.extend([data[0], data[1], data[2], data[3]])
            #
            # emotion_labels = [data[i] for i in range(4,6) if i < len(data) and data[i] is not None]
            # emotion_labels_ids = [emotion_id[e] for e in emotion_labels if e in emotion_id]
            # info.extend([1 if i in emotion_labels else 0 for i in range(24)])
            #
            # info.append(1 if data[6]=="positive" else 0)
            # info.append(data[7])
            # senticnet_embeddings.append(info)

        affective_embeddings = np.array(affective_embeddings,dtype=np.float32)
        # senticnet_embeddings = np.array(senticnet_embeddings,dtype=np.float32)
        #
        # embeddings = np.concatenate([affective_embeddings,senticnet_embeddings],axis=1)
        # 平均池化聚合为句子级特征
        return np.mean(affective_embeddings, axis=0).astype(np.float32)

    def _extract_bert_features_batch(self, texts):
        """提取BERT 768维上下文特征"""
        # 文本编码

        texts = [str(text) if text is not None else "" for text in texts]

        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        ).to(self.device)

        # 获取BERT隐藏状态

        with torch.no_grad():
            outputs = self.bert_model(**inputs)

            bert_features = outputs.last_hidden_state[:, 0, :]

        return bert_features.cpu().numpy()

    def extract_combined_features_batch(self, texts, batch_size=16):
        """提取AffectiveSpace+BERT拼接特征"""
        all_affective_features = []
        all_bert_features = []

        for i in range(0,len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]

            batch_affective_features = []

            for text in batch_texts:
                # 1. 提取概念
                concepts = self._extract_concepts(text)

                # 2. 提取AffectiveSpace特征（100维）
                affective_features = self._extract_affective_senticnet_features(concepts)
                batch_affective_features.append(affective_features)

            batch_affective_features = np.stack(batch_affective_features)

            # 3. 提取BERT特征（768维）

            batch_bert_features = self._extract_bert_features_batch(batch_texts)

            all_affective_features.append(batch_affective_features)
            all_bert_features.append(batch_bert_features)
        affective_features_batch = np.concatenate(all_affective_features, axis=0)
        bert_features_batch = np.concatenate(all_bert_features, axis=0)

        combined_features = np.concatenate([bert_features_batch,affective_features_batch], axis=1)


        return combined_features


    def get_bert_model(self):
        return self.bert_model

class FeaturesDataset(Dataset):
    def __init__(self, features,labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return {
            'features': torch.tensor(self.features[idx], dtype=torch.float),
            'labels': torch.tensor(self.labels[idx], dtype=torch.float)
        }


def read_myPersonality_data():
    url1 = "D:/datasets_1/myPersonality/困惑度/personality_perplexity.csv"
    url2 = "D:/datasets_1/myPersonality/原始数据/Personality_1.csv"
    data = pd.read_csv(
        filepath_or_buffer=url1,
        encoding='utf-8',
        header=0,
        encoding_errors='ignore',
    )

    texts = data.iloc[:,0].fillna('').astype(str).values
    labels = data.iloc[:, 1:6].astype(numpy.float16).values

    return texts,labels

def create_myPersonality_dataloaders(extractor,batch_size, texts, labels, train_idx, val_idx,):

    combined_features = extractor.extract_combined_features_batch(texts,batch_size=batch_size)


    features_train, features_val = combined_features[train_idx], combined_features[val_idx]
    labels_train, labels_val = labels[train_idx], labels[val_idx]

    train_dataset = FeaturesDataset(features_train, labels_train)
    val_dataset = FeaturesDataset(features_val, labels_val)

    # bert_model = extractor.get_bert_model()

    return (DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
            DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
            )