import random

import numpy
import numpy as np
import pandas as pd

import torch
from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset,DataLoader

short_length = 128
long_length= 512
MAX_CHUNKS = 3

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length, stride, extractor):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        self.extractor = extractor
        self.pad_token_id = tokenizer.pad_token_id

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):

        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            stride=self.stride,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            return_overflowing_tokens=True
        )

        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        num_chunks = input_ids.shape[0]

        if num_chunks < MAX_CHUNKS:
            # 填充到固定数量
            pad_num = MAX_CHUNKS - num_chunks
            pad_shape = (pad_num,self.max_length)

            # 填充input_ids (pad token ID)
            pad_ids = torch.full(pad_shape, self.pad_token_id, dtype=torch.long)
            input_ids = torch.cat((input_ids, pad_ids), dim=0)

            #填充attention_mask (用0)
            pad_mask = torch.zeros(pad_shape, dtype=torch.long)
            attention_mask = torch.cat([attention_mask, pad_mask], dim=0)
        elif num_chunks > MAX_CHUNKS:
            # 阶段到固定数量
            input_ids = input_ids[:MAX_CHUNKS]
            attention_mask = attention_mask[:MAX_CHUNKS]

        sentic_fea = self.extractor.extract_features_batch(text)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.tensor(label,dtype=torch.float32 if isinstance(label, np.ndarray) else torch.long),
            'sentic_fea': torch.from_numpy(sentic_fea[0])
        }

class TextDataset1(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length, extractor):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
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
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']
        sentic_fea = self.extractor.extract_features_batch(text)

        return {
            'input_ids': input_ids.flatten(),
            'attention_mask': attention_mask.flatten(),
            'labels': torch.tensor(label,dtype=torch.long),
            'sentic_fea': torch.from_numpy(sentic_fea[0])
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


def create_dataloaders(batch_size, texts, labels, tokenizer, max_length, stride, extractor, seed=42):

    print("划分数据集ing...")
    indices = np.arange(len(texts))
    # 先划分出60%训练集
    train_idx, temp_idx, _, temp_label = train_test_split(
        indices, labels, test_size=0.4, stratify=labels, random_state=seed
    )
    # 再划分20%验证集和测试集
    val_idx, test_idx= train_test_split(
        temp_idx, test_size=0.5, stratify=temp_label, random_state=seed
    )

    texts_train, texts_val, texts_test = texts[train_idx], texts[val_idx], texts[test_idx]
    labels_train, labels_val, labels_test = labels[train_idx], labels[val_idx], labels[test_idx]

    print("划分完毕！！！")
    print(f"训练集大小：{len(train_idx)}，验证集大小：{len(val_idx)}，测试集大小：{len(test_idx)}")

    if max_length == 512:
        train_dataset = TextDataset(texts_train, labels_train, tokenizer, max_length, stride, extractor)
        val_dataset = TextDataset(texts_val, labels_val, tokenizer, max_length, stride, extractor)
        test_dataset = TextDataset(texts_test, labels_test, tokenizer, max_length, stride, extractor)
    else:
        train_dataset = TextDataset1(texts_train, labels_train, tokenizer, max_length, extractor)
        val_dataset = TextDataset1(texts_val, labels_val, tokenizer, max_length, extractor)
        test_dataset = TextDataset1(texts_test, labels_test, tokenizer, max_length, extractor)


    generator = torch.Generator()
    generator.manual_seed(seed)

    return (DataLoader(train_dataset, batch_size=batch_size, drop_last=True, shuffle=True, generator=generator),
            DataLoader(val_dataset, batch_size=batch_size, drop_last=True, shuffle=False),
            DataLoader(test_dataset, batch_size=batch_size, drop_last=True, shuffle=False)
            )



