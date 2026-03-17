import numpy
import numpy as np
import pandas as pd

import torch

from torch.utils.data import Dataset,DataLoader

short_length = 128
long_length= 512

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
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
            'input_ids':input_ids,
            'attention_mask':attention_mask,
            'labels':torch.tensor(label,dtype=torch.long),
        }

def read_data():
    data = pd.read_csv(
        "/hy-tmp/dataset/Essays/原始数据/essays_1.csv",
        encoding='utf-8',
        header=0,
        encoding_errors='ignore',
    )

    text = data.iloc[:,0].fillna('').astype(str).values
    # labels = data.iloc[:, 1].astype(numpy.float16).values
    labels = data.iloc[:, 1:6].values
    return text,labels

def create_dataloaders(tokenizer, batch_size, texts, labels, train_idx, val_idx):
    text_train, text_val = texts[train_idx], texts[val_idx]
    labels_train, labels_val = labels[train_idx], labels[val_idx]

    train_dataset = TextDataset(text_train, labels_train, tokenizer, max_length=long_length)
    val_dataset = TextDataset(text_val, labels_val, tokenizer, max_length=long_length)
    generator = torch.Generator()
    generator.manual_seed(42)
    return (DataLoader(train_dataset, batch_size=batch_size, shuffle=True,generator=generator),
            DataLoader(val_dataset, batch_size=batch_size, shuffle=False))

def read_emo_data():
    data = pd.read_csv(
        "/hy-tmp/dataset/Essays/原始数据/essays_revise.csv",
        encoding='utf-8',
        header=0,
        encoding_errors='ignore',
    )

    text = data.iloc[:,0].fillna('').astype(str).values
    # labels = data.iloc[:, 1].astype(numpy.float16).values
    labels = data.iloc[:, 3].values
    return text,labels

def create_emo_dataloaders(tokenizer, batch_size, texts, labels, train_idx, val_idx):
    text_train, text_val = texts[train_idx], texts[val_idx]
    labels_train, labels_val = labels[train_idx], labels[val_idx]

    train_dataset = TextDataset(text_train, labels_train, tokenizer, max_length=long_length)
    val_dataset = TextDataset(text_val, labels_val, tokenizer, max_length=long_length)
    generator = torch.Generator()
    generator.manual_seed(42)
    return (DataLoader(train_dataset, batch_size=batch_size, shuffle=True,generator=generator),
            DataLoader(val_dataset, batch_size=batch_size, shuffle=False))