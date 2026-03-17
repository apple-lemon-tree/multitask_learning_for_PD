import os
import time

import torch
from torch import nn
from tqdm import tqdm

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch.optim as optim
from transformers import BertTokenizer, DistilBertTokenizer

from data_loader.load_data import *
from MultitaskModels.MyMultiTaskModel import MultitaskModel

import random
random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
print(f"Use GPU: {torch.cuda.get_device_name()}")

checkpoint_path = "essaysAug_isear_training_checkpoint.pth"
start_epoch = 0

scaler = torch.cuda.amp.GradScaler()
model_name = "../models/bert-base-uncased"
tokenizers = BertTokenizer.from_pretrained(model_name)

batch_size = 32
pnd_train_dataloader,pnd_val_dataloader = load_Essays_data_augment(tokenizers,batch_size)
isear_train_dataloader, isear_val_dataloader = load_isear_data(tokenizers,batch_size)
mbti_train_dataloader, mbti_val_dataloader = load_mbti_data(tokenizers, batch_size)
essays_train_dataloader, essays_val_dataloader = load_Essays_data_augment(tokenizers, batch_size)

isear_criterion = nn.CrossEntropyLoss().to(device)
pnd_criterion = nn.BCEWithLogitsLoss().to(device)

pos_weight = torch.tensor([3.34,6.24,1.18,1.57],dtype=torch.float).to(device)

essays_criterion = nn.BCEWithLogitsLoss().to(device)
mbti_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(device)

model = MultitaskModel()
model.to(device)

parameters = list(model.parameters())
optimizer = optim.AdamW(parameters,lr=5e-5, weight_decay=0.01, eps=1e-8)

if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])  # 加载模型参数
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  # 加载优化器状态
    scaler.load_state_dict(checkpoint['scaler_state_dict'])  # 加载混合精度训练状态
    start_epoch = checkpoint['epoch'] + 1  # 从上次结束的 epoch + 1 开始
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}, resuming training from epoch {start_epoch}")
else:
    print("No checkpoint found, starting training from scratch")

isear_iter = iter(isear_train_dataloader)
essays_iter = iter(essays_train_dataloader)
for epoch in range(start_epoch,50):

    loss_isear, loss_essays = 0, 0

    torch.cuda.empty_cache()

    torch.set_grad_enabled(True)
    print(f"Epoch:{epoch + 1}")

    model.train()
    total_batches = max(len(isear_train_dataloader),len(essays_train_dataloader))

    for i in tqdm(range(total_batches),desc="Train..."):

        try:
            isear_batch = next(isear_iter)
        except StopIteration:
            isear_iter = iter(isear_train_dataloader)
            isear_batch = next(isear_iter)
        input_ids_1 = isear_batch['input_ids'].to(device)
        attention_mask_1 = isear_batch['attention_mask'].to(device)
        labels_1 = isear_batch['labels'].to(device)

        task_1 = isear_batch['task']
        with torch.cuda.amp.autocast():
            isear_logits = model(input_ids_1,attention_mask_1,task_1)
            loss_1 = isear_criterion(isear_logits,labels_1)
        loss_isear += loss_1.item()

        try:
            essays_batch = next(essays_iter)
        except StopIteration:
            essays_iter = iter(essays_train_dataloader)
            essays_batch = next(essays_iter)
        input_ids_2 = essays_batch['input_ids'].to(device)
        attention_mask_2 = essays_batch['attention_mask'].to(device)
        labels_2 = essays_batch['labels'].to(device)
        task_2 = essays_batch['task']
        with torch.cuda.amp.autocast():
            essays_logits = model(input_ids_2, attention_mask_2, task_2)
            loss_2 = essays_criterion(essays_logits, labels_2.float())
        loss_essays += loss_2.item()
        loss = loss_1 * 0.1  + loss_2 * 0.9


        optimizer.zero_grad()

        scaler.scale(loss).backward()
        nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.0,norm_type=2)
        scaler.step(optimizer)
        scaler.update()

    print("损失情况：",loss_isear,loss_essays)


    with torch.no_grad():
        model.eval()

        isear_pred = []
        isear_label = []
        essays_pred = []
        essays_label = []
        for batch in tqdm(isear_val_dataloader, desc="isear_Eval"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            task = batch['task']
            logits = model(input_ids, attention_mask, task)
            predictions = torch.argmax(logits,dim=1)
            isear_pred.extend(predictions.cpu().numpy().flatten())
            isear_label.extend(labels.cpu().numpy().flatten())
            # print("type:",type(references),type(predictions))# Tensor Tensor


        for batch in tqdm(essays_val_dataloader, desc="essays_Eval"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            task = batch['task']
            logits = model(input_ids, attention_mask, task)
            sigmoid = nn.Sigmoid()
            probs = sigmoid(logits)
            predictions = (probs > 0.5).to(torch.int8)
            essays_pred.extend(predictions.cpu().numpy().flatten())
            essays_label.extend(labels.cpu().numpy().flatten())


        isear_result = {
            'accuracy': accuracy_score(isear_label,isear_pred),
            'precision': precision_score(isear_label,isear_pred,average='macro'),
            'recall': recall_score(isear_label,isear_pred,average='macro'),
            'f1': f1_score(isear_label,isear_pred,average='macro'),
        }


        essays_result = {
            'accuracy': accuracy_score(essays_label,essays_pred),
            'precision': precision_score(essays_label,essays_pred),
            'recall': recall_score(essays_label,essays_pred),
            'f1': f1_score(essays_label,essays_pred),
        }
        print(f"ISEAR task: {isear_result}")
        print(f"essays task: {essays_result}")

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
    }, checkpoint_path)
    print(f"Checkpoint saved for epoch {epoch + 1}")
