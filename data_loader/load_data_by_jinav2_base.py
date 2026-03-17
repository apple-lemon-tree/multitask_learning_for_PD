import numpy
import numpy as np
import pandas as pd
import spacy
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
        self.dataset_name = dataset_name

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

        input_ids = encoding['input_ids']
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

    labels = None

    if dataset_name == "myPersonality" or dataset_name == "essays":
        labels = data.iloc[:, 1:6].astype(numpy.float16).values
    elif dataset_name == "isear":
        labels = data.iloc[:, 1].astype(numpy.int8).values
    elif dataset_name == "mbti":
        labels = data.iloc[:, 1:5].astype(numpy.float16).values

    return text,labels

def create_dataloaders(tokenizer, batch_size, max_length, dataset_name, text, labels, train_idx, val_idx):
    text_train, text_val = text[train_idx], text[val_idx]
    labels_train, labels_val = labels[train_idx], labels[val_idx]

    train_dataset = TextDataset(text_train, labels_train, tokenizer, max_length=max_length)
    val_dataset = TextDataset(text_val, labels_val, tokenizer, max_length=max_length)

    return (DataLoader(train_dataset, batch_size=batch_size, drop_last=True, shuffle=True),
            DataLoader(val_dataset, batch_size=batch_size, drop_last=True, shuffle=False))
