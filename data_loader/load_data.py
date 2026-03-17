import random
import time
from collections import Counter

import numpy as np
import pandas as pd
import spacy
import torch


from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset,DataLoader
max_length = 512

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=max_length,task=None):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.task = task

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
            'task': self.task
        }

class TruncatedPlusRandomDataset(Dataset):
    def __init__(self, text, labels, tokenizer, max_len=max_length, task=None):
        self.tokenizer = tokenizer
        self.text = text
        self.labels = labels
        self.max_len = max_len
        self.task = task

    def __len__(self):
        return len(self.text)

    def select_random_sents(self, text):
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)
        sents = list(doc.sents)
        running_length = 0
        sent_idxs = list(range(len(sents)))
        selected_idx = []
        while running_length <= (self.max_len - 2) and sent_idxs:
            idx = random.choice(sent_idxs)
            sent_idxs.remove(idx)
            sentence = str(sents[idx])
            sentence_tokens = self.tokenizer.tokenize(sentence)
            running_length += len(sentence_tokens)
            selected_idx.append(idx)

        reorder_idx = sorted(selected_idx)
        selected_text = ''
        for idx in reorder_idx:
            selected_text += str(sents[idx]) + ' '

        return selected_text

    def __getitem__(self, index):
        text = str(self.text[index])
        text = str(text).strip() if text else ''
        inputs = self.tokenizer.encode_plus(
            text=text,
            text_pair=None,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            padding='max_length',
            return_attention_mask=True,
            return_token_type_ids=True,
            return_overflowing_tokens=True
        )

        if inputs.get("overflowing_tokens"):
            # select random sentences if text is longer than max length
            selected_text = self.select_random_sents(text)
            second_inputs = self.tokenizer.encode_plus(
                text=selected_text,
                text_pair=None,
                add_special_tokens=True,
                max_length=self.max_len,
                truncation=True,
                padding='max_length',
                return_attention_mask=True,
                return_token_type_ids=True,
                return_overflowing_tokens=True
            )
        else:
            second_inputs = inputs

        ids = (inputs['input_ids'], second_inputs['input_ids'])
        mask = (inputs['attention_mask'], second_inputs['attention_mask'])
        print("ids.shape:",ids)

        return {
            'input_ids': torch.tensor(ids),
            'attention_mask': torch.tensor(mask),
            'labels': torch.tensor(self.labels[index],dtype=torch.long),
            'task': self.task
        }


def load_pnd_data(tokenizers,batch_size):
    task_data = pd.read_csv(
        "../myPersonality/原始数据/Personality_1.csv",
        encoding='utf-8',
        header=0,
        encoding_errors='ignore',
    )
    pnd_text = task_data.iloc[:, 0].values
    pnd_labels = task_data.iloc[:, 1:6].values

    label_distribution = Counter(tuple(label) for label in pnd_labels)
    label_to_id = {}
    for index, (label, count) in enumerate(label_distribution.most_common()):
        label_to_id[label] = index
    new_labels = [label_to_id[tuple(label)] for label in pnd_labels]

    pnd_text_train, pnd_text_val, pnd_labels_train, pnd_labels_val = train_test_split(
        pnd_text, pnd_labels,train_size=0.7, random_state=42)
    pnd_dataset_train = TextDataset(pnd_text_train, pnd_labels_train, tokenizers, task="pnd")
    pnd_dataset_val = TextDataset(pnd_text_val, pnd_labels_val, tokenizers, task="pnd")
    pnd_train_dataloader = DataLoader(pnd_dataset_train, batch_size=batch_size, shuffle=True)
    pnd_val_dataloader = DataLoader(pnd_dataset_val, batch_size=batch_size, shuffle=False)
    return pnd_train_dataloader,pnd_val_dataloader

def load_pnd_data2(tokenizers,batch_size):
    task_data = pd.read_csv(
        "D:/datasets_1/myPersonality/困惑度小于200的样本/原数据.csv",
        encoding='utf-8',
        header=0,
        encoding_errors='ignore',
    )
    pnd_text = task_data.iloc[:, 0].values
    pnd_labels = task_data.iloc[:, 1:6].values

    # label_distribution = Counter(tuple(label) for label in pnd_labels)
    # label_to_id = {}
    # for index, (label, count) in enumerate(label_distribution.most_common()):
    #     label_to_id[label] = index
    # new_labels = [label_to_id[tuple(label)] for label in pnd_labels]

    pnd_text_train, pnd_text_val, pnd_labels_train, pnd_labels_val = train_test_split(
        pnd_text, pnd_labels,train_size=0.7, stratify=pnd_labels, random_state=42)
    pnd_dataset_train = TextDataset(pnd_text_train, pnd_labels_train, tokenizers, task="pnd")
    pnd_dataset_val = TextDataset(pnd_text_val, pnd_labels_val, tokenizers, task="pnd")
    pnd_train_dataloader = DataLoader(pnd_dataset_train, batch_size=batch_size, shuffle=True)
    pnd_val_dataloader = DataLoader(pnd_dataset_val, batch_size=batch_size, shuffle=False)
    return pnd_train_dataloader,pnd_val_dataloader

def load_pnd_data_trans(tokenizers,batch_size):
    task_data = pd.read_csv(
        "../myPersonality/回译/Personality_1.csv",
        encoding='latin1',
        header=0,
        encoding_errors='ignore',
    )
    task_data.iloc[:,0] = task_data.iloc[:,0].fillna('').astype(str)
    pnd_text = task_data.iloc[:, 0].values
    pnd_labels = task_data.iloc[:, 1:6].values

    task_data.iloc[:, 6] = task_data.iloc[:, 6].fillna('').astype(str)
    task_data.iloc[:, 7] = task_data.iloc[:, 7].fillna('').astype(str)
    en2fr = task_data.iloc[:,6].values
    en2zh = task_data.iloc[:,7].values


    (pnd_text_train, pnd_text_val,
     pnd_labels_train, pnd_labels_val,
     en2fr_train, en2fr_val,
     en2zh_train, en2zh_val) = train_test_split(
        pnd_text,
        pnd_labels,
        en2fr,
        en2zh,
        train_size=0.7,
        stratify=pnd_labels,
        random_state=42)

    new_text = np.concatenate((pnd_text_train,en2fr_train,en2zh_train),axis=0)
    new_label = np.concatenate((pnd_labels_train,pnd_labels_train,pnd_labels_train),axis=0)


    pnd_dataset_train = TextDataset(new_text, new_label, tokenizers, 128, task="pnd")
    pnd_dataset_val = TextDataset(pnd_text_val, pnd_labels_val, tokenizers, 128, task="pnd")

    pnd_train_dataloader = DataLoader(pnd_dataset_train, batch_size=batch_size, shuffle=True)
    pnd_val_dataloader = DataLoader(pnd_dataset_val, batch_size=batch_size, shuffle=False)
    return pnd_train_dataloader,pnd_val_dataloader

def load_pnd_data_augment(tokenizers,batch_size):
    train_data = pd.read_csv(
        "../myPersonality/原始数据/train_aug.csv",
        encoding='utf-8',
        encoding_errors='ignore',
        header=0,
    )
    test_data = pd.read_csv(
        "../myPersonality/原始数据/test.csv",
        encoding='utf-8',
        encoding_errors='ignore',
        header=0,
    )
    train_text = train_data.iloc[:, 0].values
    train_labels = train_data.iloc[:,1:6].values
    test_text = test_data.iloc[:, 0].values
    test_labels = test_data.iloc[:, 1:6].values

    train_dataset = TextDataset(train_text, train_labels, tokenizers, task='pnd')
    test_dataset = TextDataset(test_text, test_labels, tokenizers,task="pnd")
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, test_dataloader

def load_ESSAYS_data(tokenizers,batch_size):
    task_data = pd.read_csv(
        "../Essays/原始数据/essays_2.csv",
        encoding='utf-8',
        header=0,
    )
    pnd_text = task_data.iloc[:, 0].values
    pnd_labels = task_data.iloc[:, 1:6].values
    pnd_text_train, pnd_text_val, pnd_labels_train, pnd_labels_val = train_test_split(
        pnd_text, pnd_labels,train_size=0.7, random_state=42)
    begin = time.time()
    pnd_dataset_train = TextDataset(pnd_text_train, pnd_labels_train, tokenizers, task="essays")
    pnd_dataset_val = TextDataset(pnd_text_val, pnd_labels_val, tokenizers, task="essays")
    pnd_train_dataloader = DataLoader(pnd_dataset_train, batch_size=batch_size, shuffle=True)
    pnd_val_dataloader = DataLoader(pnd_dataset_val, batch_size=batch_size, shuffle=False)
    return pnd_train_dataloader,pnd_val_dataloader

def load_Essays_data_augment(tokenizers,batch_size):
    train_data = pd.read_csv(
        "../Essays/EDA_aug/train_aug.csv",
        encoding='utf-8',
        encoding_errors='ignore',
        header=0,
    )
    test_data = pd.read_csv(
        "../Essays/EDA_aug/test.csv",
        encoding='utf-8',
        encoding_errors='ignore',
        header=0,
    )
    train_text = train_data.iloc[:, 0].values
    train_labels = train_data.iloc[:,1:6].values
    test_text = test_data.iloc[:, 0].values
    test_labels = test_data.iloc[:, 1:6].values

    train_dataset = TextDataset(train_text, train_labels, tokenizers, task='essays')
    test_dataset = TextDataset(test_text, test_labels, tokenizers,task="essays")
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_dataloader,test_dataloader

# def load_tec_data(tokenizers,batch_size):
#     task_data = pd.read_csv(
#         "D:/datasets_1/TEC/原始数据/TEC_1.csv",
#         encoding='latin1',
#         header=0,
#     )
#     emo_text = task_data.iloc[:, 0].values
#     emo_labels = task_data.iloc[:, 1].values
#     emo_text_train, emo_text_val, emo_labels_train, emo_labels_val = train_test_split(
#         emo_text, emo_labels,train_size=0.7, random_state=42)
#     emo_dataset_train = TextDataset(emo_text_train, emo_labels_train, tokenizers,task="tec")
#     emo_dataset_val = TextDataset(emo_text_val, emo_labels_val, tokenizers, task="tec")
#     emo_train_dataloader = DataLoader(emo_dataset_train, batch_size=batch_size, shuffle=True)
#     emo_val_dataloader = DataLoader(emo_dataset_val, batch_size=batch_size, shuffle=False)
#     return emo_train_dataloader,emo_val_dataloader

def load_isear_data(tokenizer,batch_size):
    task_data = pd.read_csv(
        "../ISEAR/原始数据/ISEAR_1.csv",
        encoding='utf-8',
        encoding_errors='ignore',
        header=0,
    )
    emo_text = task_data.iloc[:, 0].values
    emo_labels = task_data.iloc[:, 1].values
    emo_text_train, emo_text_val, emo_labels_train, emo_labels_val = train_test_split(
        emo_text, emo_labels,train_size=0.7, random_state=42)
    emo_dataset_train = TextDataset(emo_text_train, emo_labels_train, tokenizer,128, task="isear")
    emo_dataset_val = TextDataset(emo_text_val, emo_labels_val, tokenizer, 128,task="isear")
    emo_train_dataloader = DataLoader(emo_dataset_train, batch_size=batch_size, shuffle=True)
    emo_val_dataloader = DataLoader(emo_dataset_val, batch_size=batch_size, shuffle=False)
    return emo_train_dataloader,emo_val_dataloader

def load_isear_data_trans(tokenizer,batch_size):
    task_data = pd.read_csv(
        "../ISEAR/回译/ISEAR_1.csv",
        encoding='utf-8',
        encoding_errors='ignore',
        header=0,
    )
    emo_text = task_data.iloc[:, 0].values
    emo_labels = task_data.iloc[:, 1].values
    en2fr = task_data.iloc[:, 2].values
    en2zh = task_data.iloc[:, 3].values


    (emo_text_train, emo_text_val,
     emo_labels_train, emo_labels_val,
     en2fr_train, en2fr_val,
     en2zh_train, en2zh_val
     ) = train_test_split(
        emo_text,
        emo_labels,
        en2fr,
        en2zh,
        train_size=0.7,
        stratify=emo_labels,
        random_state=42)


    new_text_train = np.concatenate((emo_text_train, en2fr_train,en2zh_train),axis=0)
    new_label_train = np.concatenate((emo_labels_train, emo_labels_train,emo_labels_train),axis=0)

    emo_dataset_train = TextDataset(new_text_train, new_label_train, tokenizer,128, task="isear")
    emo_dataset_val = TextDataset(emo_text_val, emo_labels_val, tokenizer, 128,task="isear")
    emo_train_dataloader = DataLoader(emo_dataset_train, batch_size=batch_size, shuffle=True)
    emo_val_dataloader = DataLoader(emo_dataset_val, batch_size=batch_size, shuffle=False)
    return emo_train_dataloader,emo_val_dataloader

# def load_isear_data_augment(tokenizers,batch_size):
#     train_data = pd.read_csv(
#         "D:/datasets_1/ISEAR/原始数据/train_aug.csv",
#         encoding='utf-8',
#         encoding_errors='ignore',
#         header=0,
#     )
#
#     test_data = pd.read_csv(
#         "D:/datasets_1/ISEAR/原始数据/test.csv",
#         encoding='utf-8',
#         encoding_errors='ignore',
#         header=0,
#     )
#     train_text = train_data.iloc[:, 0].values
#     train_labels = train_data.iloc[:, 1].values
#
#     test_text = test_data.iloc[:, 0].values
#     test_label = test_data.iloc[:, 1].values
#     emo_dataset_train = TextDataset(train_text, train_labels, tokenizers,task="isear")
#     emo_dataset_val = TextDataset(test_text, test_label, tokenizers, task="isear")
#     emo_train_dataloader = DataLoader(emo_dataset_train, batch_size=batch_size, shuffle=True)
#     emo_val_dataloader = DataLoader(emo_dataset_val, batch_size=batch_size, shuffle=False)
#     return emo_train_dataloader,emo_val_dataloader

def load_mbti_data(tokenizers,batch_size):
    task_data = pd.read_csv(
        "../MBTI/原始数据/kaggle_mbti.csv",
        encoding='utf-8',
        encoding_errors='ignore',
        header=0,
    )
    mbti_text = task_data.iloc[:, 0].values
    mbti_labels = task_data.iloc[:, 1:5].values
    mbti_text_train, mbti_text_val, mbti_labels_train, mbti_labels_val = train_test_split(
        mbti_text, mbti_labels,train_size=0.7, random_state=42)
    mbti_dataset_train = TextDataset(mbti_text_train, mbti_labels_train, tokenizers,task="mbti")
    mbti_dataset_val = TextDataset(mbti_text_val, mbti_labels_val, tokenizers,task="mbti")
    mbti_train_dataloader = DataLoader(mbti_dataset_train, batch_size=batch_size, shuffle=True)
    mbti_val_dataloader = DataLoader(mbti_dataset_val, batch_size=batch_size, shuffle=False)
    return mbti_train_dataloader,mbti_val_dataloader
