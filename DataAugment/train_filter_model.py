import argparse
import json
import os
import random
import re

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score,f1_score

from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup

MAX_CHUNKS = 30

def hyperparameters():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", default="Kaggle",type=str)
    parser.add_argument("--train_epochs", default=30, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--learning_rate", default=1e-5, type=float)
    parser.add_argument("--max_len", default=64, type=int)
    parser.add_argument("--model_url", default="/hy-tmp/models/bert-base-uncased", type=str)

    args = parser.parse_args()
    if args.dataset == "Essays":
        args.data_url = f"/hy-tmp/Essays/essays_revise.csv"
    elif args.dataset == "Kaggle":
        args.data_url = f"/hy-tmp/Kaggle/kaggle_original(8label).csv"
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.filter_save_url = f"filter_model/{args.dataset}"
    return args

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class PersonalityDataset(Dataset):
    def __init__(self, encodings, labels):
        self.input_ids = encodings["input_ids"]
        self.attention_mask = encodings["attention_mask"]
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        return{
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "label": torch.tensor(self.labels[idx], dtype=torch.float),
        }

class MBTIDateset(Dataset):
    def __init__(self,posts, labels, tokenizer, max_len):
        self.posts = posts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.posts)

    def __getitem__(self, idx):
        post = self.posts[idx]
        label = self.labels[idx]
        sentences = post.split("|||")[:MAX_CHUNKS]

        encodings = self.tokenizer(
            sentences,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
        )

        num_current_sentences = encodings["input_ids"].size(0)
        if num_current_sentences < MAX_CHUNKS:
            pad_chunk_num = MAX_CHUNKS - num_current_sentences

            encodings['input_ids'] = torch.cat([
                encodings['input_ids'],
                torch.zeros((pad_chunk_num, self.max_len), dtype=torch.long)
            ], dim=0)
            encodings['attention_mask'] = torch.cat([
                encodings['attention_mask'],
                torch.zeros((pad_chunk_num, self.max_len), dtype=torch.long)
            ], dim=0)

        return {
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask'],
            'label': torch.tensor(label,dtype=torch.float32),
        }


def collate_fn(batch):

    input_ids, attention_mask, label = [],[],[]
    for item in batch:
        num_chunks = len(item['input_ids'])
        ids_cat = [chunk for chunk in item['input_ids']]
        mask_cat = [chunk for chunk in item['attention_mask']]

        ids_cat = torch.stack(ids_cat,dim=0)
        mask_cat = torch.stack(mask_cat,dim=0)

        input_ids.append(ids_cat)
        attention_mask.append(mask_cat)
        label.append(item['label'])

    input_ids = torch.stack(input_ids,dim=0)
    attention_mask = torch.stack(attention_mask,dim=0)
    label = torch.stack(label)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "label": label,
    }

class Filter(nn.Module):
    def __init__(self, class_num,model_url):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_url)
        self.class_num = class_num
        self.classifier = nn.Linear(768,class_num)

    def forward(self,input_ids, attention_mask):
        if self.class_num == 4:
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            cls = outputs.last_hidden_state[:,0,:]
            pred_list = self.classifier(cls)
            return pred_list
        elif self.class_num == 8:
            if input_ids.dim() == 3:
                # print(f"input_ids.shape={input_ids.shape}") (batch_size, MAX_CHUNKS, max_len)
                batch_size, num_chunks, seq_length = input_ids.size()
                input_ids = input_ids.view(-1, seq_length)
                attention_mask = attention_mask.view(-1, seq_length)
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            pooled_output = outputs.last_hidden_state[:,0,:]
            pooled_output = pooled_output.view(batch_size, num_chunks, -1)
            pooled_output = pooled_output.mean(dim=1)

            pred_list = self.classifier(pooled_output)
            # print(f"pred_list.shape={pred_list.shape}") (batch_size, 8)
            return pred_list

def load_data(args):
    data = pd.read_csv(
        filepath_or_buffer=args.data_url,
        header=0,
        encoding="utf-8",
        encoding_errors="ignore",
    )
    if args.dataset == "Essays":
        texts = data["text"].tolist()
        labels = data[["EXT","NEU","AGR","CON","OPN"]].values.tolist()
    elif args.dataset == "Kaggle":
        texts = data["posts"].tolist()
        labels = data[["introversion", "extraversion", "intuition", "sensing", "feeling", "thinking", "perception", "judgment"]].values.tolist()
        texts = [kaggle_data_processing(text) for text in texts]
    return texts, labels

def kaggle_data_processing(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = text.encode('ascii', 'ignore').decode('ascii')
    if text.startswith("'"):
        text = text[1:-1]

    MBTIs = ('INTJ', 'INTP', 'INFP', 'ENTP', 'ISTP', 'ISFP', 'ESTJ', 'ISTJ',
             'ESTP', 'ISFJ', 'ENFP', 'ESFP', 'ESFJ', 'ENFJ', 'INFJ', 'ENTJ')
    token = '<mask>'
    for mbti in MBTIs:
        if mbti in text:
            text = text.replace(mbti.lower(), token)

    return text

def run(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_url)
    texts, labels = load_data(args)
    train_texts,test_texts,train_labels,test_labels = train_test_split(
        texts,
        labels,
        test_size=0.2,
        random_state=args.seed,
        shuffle=True,
        stratify=labels)
    if args.dataset == "Essays":
        train_encodings = tokenizer(
            train_texts,
            truncation=True,
            max_length=args.max_len,
            padding="max_length",
            return_tensors="pt",
        )
        test_encodings = tokenizer(
            test_texts,
            truncation=True,
            max_length=args.max_len,
            padding="max_length",
            return_tensors="pt",
        )
        train_dataset = PersonalityDataset(train_encodings, train_labels)
        test_dataset = PersonalityDataset(test_encodings, test_labels)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    elif args.dataset == "Kaggle":
        train_dataset = MBTIDateset(train_texts,train_labels,tokenizer,args.max_len)
        test_dataset = MBTIDateset(test_texts,test_labels,tokenizer,args.max_len)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,collate_fn=collate_fn)

    class_num = 5 if args.dataset == "Essays" else 8

    model = Filter(class_num,args.model_url).to(args.device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate,weight_decay=0.01)

    total_steps = len(train_loader) * args.train_epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    best_macro_f1 = 0
    for epoch in range(args.train_epochs):
        print(f"Epoch: {epoch+1}/{args.train_epochs}")
        avg_loss = train_epoch(model, train_loader, criterion, optimizer, scheduler,args.device)
        eval_metrics = evaluate(model,test_loader,args.device)

        print(f'训练总损失: {avg_loss:.4f}')
        print(f'Accuracy: {eval_metrics["accuracy"]:.4f}, Macro-F1: {eval_metrics["macro-f1"]:.4f}')

        if eval_metrics["macro-f1"] > best_macro_f1:
            best_macro_f1 = eval_metrics["macro-f1"]
            if not os.path.exists(args.filter_save_url):
                os.makedirs(args.filter_save_url,exist_ok=True)
            torch.save(
                model.state_dict(),
                args.filter_save_url+"/best_filter.pt")
            print(f'最好模型的macro-f1: {best_macro_f1:.4f}')

    final_metrics = evaluate(model, test_loader, args.device)
    print(f"Accuracy: {final_metrics['accuracy']:.4f}")
    print(f"Macro F1: {final_metrics['macro-f1']:.4f}")


def train_epoch(model,dataloader,criterion,optimizer,scheduler,device):
    model.train()
    total_loss = 0.0

    progress_bar = tqdm(dataloader, desc="Training")
    for _, batch in enumerate(progress_bar):
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        output = model(input_ids, attention_mask)
        loss = criterion(output, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
    return total_loss / len(dataloader)


def evaluate(model,test_loader,device):
    model.eval()
    predict, labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            label = batch["label"].to(device)
            output = model(input_ids, attention_mask)
            probs = torch.sigmoid(output)
            predictions = (probs > 0.5).int()
            predict.append(predictions.cpu())
            labels.append(label.cpu())

    predict = torch.cat(predict).numpy()
    labels = torch.cat(labels).numpy()
    # print(predict)
    # print(labels)
    metrics = {
        'accuracy': accuracy_score(labels.flatten(), predict.flatten()),
        'macro-f1': f1_score(labels, predict, average='macro'),
    }
    return metrics

if __name__ == '__main__':
    args = hyperparameters()
    set_seed(args.seed)
    run(args)

