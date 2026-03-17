import random

import numpy
import numpy as np
import pandas as pd
import spacy
import torch

from torch.utils.data import Dataset,DataLoader

short_length = 128
long_length= 512

try:
    NLP = spacy.load("en_core_web_sm")
except OSError:
    print("SpaCy 模型 'en_core_web_sm' 未找到。请运行: python -m spacy download en_core_web_sm")
    raise

# 设置一个高效的停用词集合 (Set) 用于快速查找
SPACY_STOP_WORDS = NLP.Defaults.stop_words


def remove_stopwords_spacy(text: str) -> str:
    """
    使用 spaCy 对文本进行分词、去除停用词和标点，并返回清理后的文本。
    """
    if not text:
        return ""

    # 使用 nlp.pipe() 处理大批文本效率更高
    doc = NLP(text)

    # 过滤停用词、标点和空格
    # token.is_stop: 检查是否为停用词
    # token.is_punct: 检查是否为标点符号
    # token.text.strip(): 移除空格，防止留下空字符串

    filtered_words = [
        token.lemma_.lower()  # 使用词形还原 (lemma) 效果更好，并转小写
        for token in doc
        if not token.is_stop and
           not token.is_punct and
           token.text.strip()  # 确保不是空字符串
    ]

    if len(filtered_words) == 0:
        print("去掉停止词后为空")
    return " ".join(filtered_words)



class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length


    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):

        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            add_special_tokens=False,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].flatten()
        attention_mask = encoding['attention_mask'].flatten()
        return {
            'texts': text,
            'input_ids':input_ids,
            'attention_mask':attention_mask,
            'labels':torch.tensor(label,dtype=torch.long),
        }

def read_data(url,dataset_name):
    data = pd.read_csv(
        filepath_or_buffer=url,
        encoding='utf-8',
        header=0,
        encoding_errors='ignore',
    )
    text = data.iloc[:, 0].fillna('').astype(str).values
    # cleaned_texts = np.array([remove_stopwords_spacy(t) for t in text])


    labels = None

    if dataset_name == "myPersonality" or dataset_name == "essays":
        labels = data.iloc[:, 1:6].astype(numpy.float16).values
    elif dataset_name == "isear":
        labels = data.iloc[:, 1].astype(numpy.int8).values
    elif dataset_name == "mbti":
        labels = data.iloc[:, 1:5].astype(numpy.float16).values

    return text,labels

def worker_init_fn():
    """为每个数据加载器工作进程设置随机种子"""
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    numpy.random.seed(worker_seed)
    torch.manual_seed(worker_seed)

def load_AugmentedData(url):
    data = pd.read_csv(
        url,
        header=0,
        encoding='utf-8',
        encoding_errors='ignore',
    )

    texts = data.iloc[:, 0].fillna('').astype(str).values
    labels = data.iloc[:, 1:6].astype(numpy.int8).values

    return texts,labels

def create_dataloaders(tokenizer, batch_size, max_length, text, labels, train_idx, val_idx, seed=42):
    text_train, text_val = text[train_idx], text[val_idx]
    labels_train, labels_val = labels[train_idx], labels[val_idx]

    """
        text_train += GPT生成的数据 + 回译数据
        labels_train += GPT生成的数据 + 回译数据
        """
    zh_url = "/hy-tmp/dataset/Essays/回译/zh_bt_essays.csv"
    fr_url = "/hy-tmp/dataset/Essays/回译/fr_bt_essays.csv"
    aug_url = "/hy-tmp/dataset/Essays/GPT生成/ori_aug.csv"
    # 中文回译
    zh_text, zh_label = load_AugmentedData(zh_url)
    fr_text, fr_label = load_AugmentedData(fr_url)
    aug_text, aug_label = load_AugmentedData(aug_url)

    for i in range(len(zh_text)):
        if zh_text[i] == "999" or fr_text[i] == "999":
            continue
        text_train = np.concatenate((text_train, zh_text[i], fr_text[i]))
        labels_train = np.concatenate((labels_train, zh_label[i], fr_label[i]), axis=0)

    text_train = np.concatenate((text_train, aug_text))
    labels_train = np.concatenate((labels_train, aug_label), axis=0)

    train_dataset = TextDataset(text_train, labels_train, tokenizer, max_length=max_length)
    val_dataset = TextDataset(text_val, labels_val, tokenizer, max_length=max_length)

    generator = torch.Generator()
    generator.manual_seed(seed)

    return (DataLoader(train_dataset, batch_size=batch_size, drop_last=True, shuffle=True,generator=generator, worker_init_fn=worker_init_fn()),
            DataLoader(val_dataset, batch_size=batch_size, drop_last=True, shuffle=False))
