import random
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error
from torch import nn
from tqdm import tqdm
import torch.optim as optim
from transformers import AutoTokenizer, BertTokenizer

from data_loader.load_data import *
from MultitaskModels.model3 import MultitaskModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
scaler = torch.cuda.amp.GradScaler()

model_name = "../models/bert-base-uncased"
tokenizers = BertTokenizer.from_pretrained(model_name)

batch_size = 2
pnd_train_dataloader,pnd_val_dataloader = load_ESSAYS_data(tokenizers,batch_size)
mbti_train_dataloader, mbti_val_dataloader = load_mbti_data(tokenizers, batch_size)

pnd_loss_func = nn.BCEWithLogitsLoss().to(device)
mbti_loss_func = nn.BCEWithLogitsLoss().to(device)

model = MultitaskModel()


model.to(device)
parameters = list(model.parameters())

optimizer = optim.AdamW(parameters,lr=1e-5,weight_decay=0.01,eps=1e-8)
# scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
loss_pnd, loss_isear, loss_mbti = 10000, 10000, 10000

random.seed(42)
torch.manual_seed(42)

for epoch in range(50):

    torch.cuda.empty_cache()

    print(f"Epoch:{epoch + 1}")

    model.train()

    total_train_batch = []
    for batch in pnd_train_dataloader:
        total_train_batch.append(batch)

    for batch in mbti_train_dataloader:
        total_train_batch.append(batch)

    random.shuffle(total_train_batch)

    pnd_loss, mbti_loss = 0, 0
    for batch in tqdm(total_train_batch,desc="Train"):
        with torch.cuda.amp.autocast():
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            task = batch['task']
            logits = model(input_ids,attention_mask,task)
            loss = None
            if 'essays' in task:
                loss = pnd_loss_func(logits,labels.float())
                pnd_loss += loss.item()

            elif 'mbti' in task:
                loss = mbti_loss_func(logits,labels.float())
                mbti_loss += loss.item()

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.0,norm_type=2)

    print("损失情况：",pnd_loss,mbti_loss)



    with torch.no_grad():
        model.eval()

        pnd_pred = []
        pnd_label = []
        pnd_logits = []

        mbti_pred = []
        mbti_label = []
        mbti_logits = []

        for batch in tqdm(pnd_val_dataloader, desc="PND_Eval"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            task = batch['task']
            logits = model(input_ids, attention_mask, task)
            sigmoid = nn.Sigmoid()
            probs = sigmoid(logits)
            pnd_logits.extend(probs.cpu().numpy().flatten())
            predictions = (probs > 0.5).to(torch.int8)
            references = labels
            pnd_pred.extend(predictions.cpu().numpy().flatten())
            pnd_label.extend(labels.cpu().numpy().flatten())
            # print("type:",type(references),type(predictions))# Tensor Tensor

        for batch in tqdm(mbti_val_dataloader, desc="MBTI_Eval"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            task = batch['task']
            logits = model(input_ids, attention_mask, task)
            sigmoid = nn.Sigmoid()
            probs = sigmoid(logits)
            mbti_logits.extend(probs.cpu().numpy().flatten())
            predictions = (probs > 0.5).to(torch.int8)
            references = labels
            mbti_pred.extend(predictions.cpu().numpy().flatten())
            mbti_label.extend(labels.cpu().numpy().flatten())


        # for batch in tqdm(mbti_val_dataloader, desc="MBTI_Eval"):
        #     input_ids = batch['input_ids'].to(device)
        #     attention_mask = batch['attention_mask'].to(device)
        #     labels = batch['labels'].to(device)
        #     task = batch['task']
        #     logits = model(input_ids, attention_mask, task)
        #     probs = torch.argmax(logits, dim=1)
        #
        #     # sigmoid = nn.Sigmoid()
        #     # probs = sigmoid(logits)
        #     # predictions = (probs > 0.5).to(torch.int8)
        #     emo_pred.extend(probs.cpu().numpy().flatten())
        #     emo_label.extend(labels.cpu().numpy().flatten())

        pnd_result = {
            'accuracy': accuracy_score(pnd_label, pnd_pred),
            'precision': precision_score(pnd_label, pnd_pred),
            'recall': recall_score(pnd_label, pnd_pred),
            'f1': f1_score(pnd_label, pnd_pred),
            'MAE': mean_absolute_error(pnd_label,pnd_logits)
        }

        mbti_result = {
            'accuracy': accuracy_score(mbti_label, mbti_pred),
            'precision': precision_score(mbti_label, mbti_pred),
            'recall': recall_score(mbti_label, mbti_pred),
            'f1': f1_score(mbti_label, mbti_pred),
        }
        print(f"Personaliaty task: {pnd_result}")
        print(f"MBTI task: {mbti_result}")
