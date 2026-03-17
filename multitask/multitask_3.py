import time

from torch import nn
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch.optim as optim
from transformers import AutoTokenizer, BertTokenizer, DistilBertTokenizer

from data_loader.load_data import *
from MultitaskModels.model3 import MultitaskModel

from myutils.weight_methods import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model_name = "../models/distilbert-base-uncased"
tokenizers = DistilBertTokenizer.from_pretrained(model_name)

batch_size = 8
pnd_train_dataloader,pnd_val_dataloader = load_Essays_data_augment(tokenizers,batch_size)
isear_train_dataloader, isear_val_dataloader = load_isear_data(tokenizers,batch_size)
mbti_train_dataloader, mbti_val_dataloader = load_mbti_data(tokenizers, batch_size)

isear_criterion = nn.CrossEntropyLoss().to(device)

pnd_criterion = nn.BCEWithLogitsLoss().to(device)
# pnd_criterion = ResampleLoss(reweight_func='rebalance', loss_weight=1.0,
#                              focal=dict(focal=True, alpha=0.5, gamma=2),
#                              logit_reg=dict(init_bias=0.05, neg_scale=2.0),
#                              map_param=dict(alpha=0.1, beta=10.0, gamma=0.9),
#                              class_freq=pnd_class_freq, train_num=6941)
mbti_criterion = nn.BCEWithLogitsLoss().to(device)
# mbti_criterion = ResampleLoss(reweight_func='rebalance', loss_weight=1.0,
#                              focal=dict(focal=True, alpha=0.5, gamma=2),
#                              logit_reg=dict(init_bias=0.05, neg_scale=2.0),
#                              map_param=dict(alpha=0.1, beta=10.0, gamma=0.9),
#                              class_freq=class_freq, train_num=train_num)

model = MultitaskModel()
model.to(device)

parameters = list(model.parameters())

# 5e-5 0.00005

optimizer = optim.AdamW(parameters,lr=5e-5)
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
weighting_method = MGDA(n_tasks=2,device=device)
loss_pnd, loss_mbti = 10000, 10000

pnd_iter = iter(pnd_train_dataloader)
mbti_iter = iter(mbti_train_dataloader)
for epoch in range(30):



    torch.set_grad_enabled(True)
    print(f"Epoch:{epoch + 1}")

    model.train()
    total_batches = max(len(pnd_train_dataloader),len(mbti_train_dataloader))

    for i in tqdm(range(total_batches),desc="Train..."):
        optimizer.zero_grad()
        try:
            personality_batch = next(pnd_iter)
        except StopIteration:
            pnd_iter = iter(pnd_train_dataloader)
            personality_batch = next(pnd_iter)
        input_ids_1 = personality_batch['input_ids'].to(device)
        attention_mask_1 = personality_batch['attention_mask'].to(device)
        labels_1 = personality_batch['labels'].to(device)

        task_1 = personality_batch['task']
        personality_logits = model(input_ids_1,attention_mask_1,task_1)
        loss_1 = pnd_criterion(personality_logits,labels_1.float())

        try:
            mbti_batch = next(mbti_iter)
        except StopIteration:
            mbti_iter = iter(mbti_train_dataloader)
            mbti_batch = next(mbti_iter)
        input_ids_2 = mbti_batch['input_ids'].to(device)
        attention_mask_2 = mbti_batch['attention_mask'].to(device)
        labels_2 = mbti_batch['labels'].to(device)
        task_2 = mbti_batch['task']
        isear_logits = model(input_ids_2, attention_mask_2, task_2)
        loss_2 = mbti_criterion(isear_logits, labels_2.float())
        losses = torch.stack((loss_1,loss_2))
        weighted_loss, extra_outputs = weighting_method.backward(
            losses=losses,
            shared_parameters=model.shared_parameters(),
            task_specific_parameters=model.task_specific_paramters()
        )
        #print(f"任务权重: {extra_outputs['weights']}")
        # loss = loss_1 * 100 + loss_2 * 100
        #
        # loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.0,norm_type=2)

        optimizer.step()


    with torch.no_grad():
        model.eval()

        pnd_pred = []
        pnd_label = []
        mbti_pred = []
        mbti_label = []
        for batch in tqdm(pnd_val_dataloader, desc="PND_Eval"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            task = batch['task']
            logits = model(input_ids, attention_mask, task)
            sigmoid = nn.Sigmoid()
            probs = sigmoid(logits)
            predictions = (probs > 0.5).to(torch.int8)
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
            predictions = (probs > 0.5).to(torch.int8)
            mbti_pred.extend(predictions.cpu().numpy().flatten())
            mbti_label.extend(labels.cpu().numpy().flatten())


        pnd_result = {
            'accuracy': accuracy_score(pnd_label,pnd_pred),
            'precision': precision_score(pnd_label,pnd_pred),
            'recall': recall_score(pnd_label,pnd_pred),
            'f1': f1_score(pnd_label,pnd_pred),
        }


        mbti_result = {
            'accuracy': accuracy_score(mbti_label,mbti_pred),
            'precision': precision_score(mbti_label,mbti_pred,average='macro'),
            'recall': recall_score(mbti_label,mbti_pred,average='macro'),
            'f1': f1_score(mbti_label,mbti_pred,average='macro'),
        }
        print(f"Personaliaty task: {pnd_result}")
        print(f"MBTI task: {mbti_result}")

    scheduler.step()

