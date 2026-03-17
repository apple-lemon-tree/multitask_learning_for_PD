import random

import numpy
import numpy as np
import pandas as pd

import torch

from torch.utils.data import Dataset,DataLoader

from myutils.SingleTextAff import SingleTextAffectiveSpace

extractor = SingleTextAffectiveSpace("/hy-tmp/dataset/affectivespace/affective.csv")
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
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
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].flatten()
        attention_mask = encoding['attention_mask'].flatten()
        sentic_fea = self.extractor.extract_features_batch(text)

        return {
            'input_ids':input_ids,
            'attention_mask':attention_mask,
            'labels':torch.tensor(label,dtype=torch.long),
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

    if dataset_name == "essays":
        labels = data.iloc[:, 1:6].astype(numpy.float16).values
    elif dataset_name == "isear":
        labels = data.iloc[:, 1].astype(numpy.int8).values
    elif dataset_name == "mbti":
        labels = data.iloc[:, 1:5].astype(numpy.float16).values
    return text,labels

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

def create_dataloaders(tokenizer, batch_size, max_length, text, labels, train_idx, val_idx,seed=67373):

    if len(text) == 8675:
        text_train, text_val = text[train_idx], text[val_idx]
        labels_train, labels_val = labels[train_idx], labels[val_idx]
        url = "/hy-tmp/dataset/MBTI/mbti_original.csv"
        aug_url = "/hy-tmp/dataset/MBTI/augmented_data.csv"
        bt_text, bt_label = load_AugmentedData(url)
        aug_text, aug_label = load_AugmentedData(aug_url)

        bt_text, bt_label = bt_text[train_idx], bt_label[train_idx]

        for i in range(len(bt_text)):
            if bt_text[i] == "low equality":
                continue
            text_train = np.concatenate((text_train, [bt_text[i]]))
            labels_train = np.concatenate((labels_train, [bt_label[i]]), axis=0)
        text_train = np.concatenate((text_train, aug_text))
        labels_train = np.concatenate((labels_train, aug_label), axis=0)
        train_dataset = TextDataset(text_train, labels_train, tokenizer, max_length)
        val_dataset = TextDataset(text_val, labels_val, tokenizer, max_length)
        print("after data augmentation")
        print(f"train data:{len(train_dataset)},test data:{len(val_dataset)}")
    else:
        text_train, text_val = text[train_idx], text[val_idx]
        labels_train, labels_val = labels[train_idx], labels[val_idx]
        train_dataset = TextDataset(text_train, labels_train, tokenizer, max_length)
        val_dataset = TextDataset(text_val, labels_val, tokenizer, max_length)
        print(f"train data：{len(train_dataset)},test data：{len(val_dataset)}")
    generator = torch.Generator()
    generator.manual_seed(seed)

    return (DataLoader(train_dataset, batch_size=batch_size, drop_last=True, shuffle=True, generator=generator),
            DataLoader(val_dataset, batch_size=batch_size, drop_last=True, shuffle=False))
