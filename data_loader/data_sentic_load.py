import random

import numpy
import numpy as np
import pandas as pd

import torch
from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset,DataLoader

short_length = 128
long_length= 512

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length,extractor):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        # self.stride = stride
        self.extractor = extractor


    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):

        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            # stride=self.stride,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            # return_overflowing_tokens=True
        )
        input_ids = encoding['input_ids'].flatten()
        attention_mask = encoding['attention_mask'].flatten()
        sentic_fea = self.extractor.extract_features_batch(text)
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.tensor(label,dtype=torch.long),
            'sentic_fea': torch.from_numpy(sentic_fea)
        }

def read_data(url,dataset_name):
    data = pd.read_csv(
        filepath_or_buffer=url,
        encoding='utf-8',
        header=0,
        encoding_errors='ignore',
    )
    text = data.iloc[:, 0].fillna('').astype(str).values

    labels = None

    if dataset_name == "myPersonality" or dataset_name == "essays":
        labels = data.iloc[:, 1:6].astype(numpy.float16).values
    elif dataset_name == "isear":
        labels = data.iloc[:, 1].astype(numpy.int8).values
    elif dataset_name == "mbti":
        labels = data.iloc[:, 1:5].astype(numpy.float16).values
    print(len(text),len(labels))
    return text,labels

def worker_init_fn():
    """为每个数据加载器工作进程设置随机种子"""
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    numpy.random.seed(worker_seed)
    torch.manual_seed(worker_seed)

def load_AugmentedData(url):
    zh_url = "/hy-tmp/dataset/Essays/回译/zh_bt_essays.csv"
    fr_url = "/hy-tmp/dataset/Essays/回译/fr_bt_essays.csv"

    data = pd.read_csv(
        url,
        header=0,
        encoding='utf-8',
        encoding_errors='ignore',
    )

    texts = data.iloc[:, 0].fillna('').astype(str).values
    labels = data.iloc[:, 1:6].astype(numpy.int8).values

    return texts,labels

def create_dataloaders_trans(tokenizer, batch_size, max_length, extractor, text, labels, train_idx, val_idx, seed=42):
    text_train, text_val = text[train_idx], text[val_idx]
    labels_train, labels_val = labels[train_idx], labels[val_idx]
    """
    text_train += GPT生成的数据 + 回译数据
    labels_train += GPT生成的数据 + 回译数据
    """
    url = "/hy-tmp/dataset/Essays/回译/en2fr2de2en_bt_essays0.csv"
    aug_url = "/hy-tmp/dataset/Essays/GPT生成/ori_aug.csv"
    # 中文回译
    bt_text, bt_label = load_AugmentedData(url)
    aug_text, aug_label = load_AugmentedData(aug_url)

    bt_text,bt_label = bt_text[train_idx], bt_label[train_idx]

    for i in range(len(bt_text)):
        if bt_text[i] == "999":
            continue
        text_train = np.concatenate((text_train,[bt_text[i]]))
        labels_train = np.concatenate((labels_train,[bt_label[i]]),axis=0)
        # labels_train = np.vstack([labels_train,zh_label[i],fr_label[i]])
    text_train = np.concatenate((text_train,aug_text))
    labels_train = np.concatenate((labels_train,aug_label),axis=0)

    train_dataset = TextDataset(text_train, labels_train, tokenizer, max_length,extractor)
    val_dataset = TextDataset(text_val, labels_val, tokenizer, max_length,extractor)
    print(f"训练数据数量：{len(train_dataset)}，测试数据数量：{len(val_dataset)}")
    generator = torch.Generator()
    generator.manual_seed(seed)

    return (DataLoader(train_dataset, batch_size=batch_size, drop_last=True, shuffle=True, generator=generator,worker_init_fn=worker_init_fn()),
            DataLoader(val_dataset, batch_size=batch_size, drop_last=True, shuffle=False))


def create_dataloaders(batch_size, texts, labels, tokenizer, max_length, stride, extractor, seed=42):
    print("划分数据集ing...")
    indices = np.arange(len(texts))
    # 先划分出60%训练集
    train_idx, temp_idx, _, temp_label = train_test_split(
        indices, labels, test_size=0.4, stratify=labels, random_state=seed
    )
    # 再划分20%验证集和测试集
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, stratify=temp_label, random_state=seed
    )

    texts_train, texts_val, texts_test = texts[train_idx], texts[val_idx], texts[test_idx]
    labels_train, labels_val, labels_test = labels[train_idx], labels[val_idx], labels[test_idx]

    print("划分完毕！！！")
    print(f"训练集大小：{len(train_idx)}，验证集大小：{len(val_idx)}，测试集大小：{len(test_idx)}")
    if labels.ndim == 2:
        url = "/hy-tmp/dataset/Essays/回译/en2fr2de2en_bt_essays0.csv"
        aug_url = "/hy-tmp/dataset/Essays/GPT生成/ori_aug.csv"
        # 回译数据和GPT生成数据
        bt_text, bt_label = load_AugmentedData(url)
        aug_text, aug_label = load_AugmentedData(aug_url)

        bt_text, bt_label = bt_text[train_idx], bt_label[train_idx]

        for i in range(len(bt_text)):
            if bt_text[i] == "999":
                continue
            texts_train = np.concatenate((texts_train, [bt_text[i]]))
            labels_train = np.concatenate((labels_train, [bt_label[i]]), axis=0)
            # labels_train = np.vstack([labels_train,zh_label[i],fr_label[i]])
        texts_train = np.concatenate((texts_train, aug_text))
        labels_train = np.concatenate((labels_train, aug_label), axis=0)

        print("人格数据的数据增强完毕！！！")
        print(f"训练集大小：{len(texts_train)}，验证集大小：{len(val_idx)}，测试集大小：{len(test_idx)}")

    train_dataset = TextDataset(texts_train, labels_train, tokenizer, max_length, extractor)
    val_dataset = TextDataset(texts_val, labels_val, tokenizer, max_length, extractor)
    test_dataset = TextDataset(texts_test, labels_test, tokenizer, max_length, extractor)

    generator = torch.Generator()
    generator.manual_seed(seed)

    return (DataLoader(train_dataset, batch_size=batch_size, drop_last=True, shuffle=True, generator=generator),
            DataLoader(val_dataset, batch_size=batch_size, drop_last=True, shuffle=False),
            DataLoader(test_dataset, batch_size=batch_size, drop_last=True, shuffle=False)
            )



