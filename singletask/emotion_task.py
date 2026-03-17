import torch
from sklearn.model_selection import StratifiedKFold
from torch import nn
from tqdm import tqdm

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch.optim as optim
from transformers import AutoTokenizer

from data_loader.load_emo import *
from SingleTaskModels.model2 import SingletaskModel
from myutils.EarlyStop import EarlyStopping

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model_name = "/hy-tmp/models/bert-base-uncased"
tokenizers = AutoTokenizer.from_pretrained(model_name)

batch_size = 16
k_fold = 5

e_text, e_labels = read_emo_data()


skf = StratifiedKFold(n_splits=k_fold, shuffle=True,random_state=42)

emotion_criterion = nn.BCEWithLogitsLoss().to(device)

Isear_K = skf.split(e_text, e_labels)

K_emo_accuracy= []

for fold, (e_train_index, e_test_index) in enumerate(Isear_K):
    print("*"*50)
    print(f"第{fold + 1}折")
    print("*" * 50)

    model = SingletaskModel(model_name).to(device)

    parameters = list(model.parameters())

    optimizer = optim.AdamW(parameters, lr=5e-6, weight_decay=0.01, eps=1e-8)

    early_stopping = EarlyStopping(patience=5, verbose=True,path="single_essyas1.pt")

    emotion_train_dataloader,emotion_val_dataloader = (
        create_emo_dataloaders(
            tokenizer=tokenizers,batch_size=batch_size,texts=e_text,
            labels=e_labels,train_idx=e_train_index,val_idx=e_test_index
        )
    )

    emo_iter = iter(emotion_train_dataloader)

    max_accuracy = 0.0

    for epoch in range(50):
        print(f"Epoch:{epoch + 1}")

        torch.cuda.empty_cache()
        model.train()

        batchs_num = len(emotion_train_dataloader)

        e_loss = 0.0
        for emotion_batch in tqdm(emotion_train_dataloader, desc="Emotion Training..."):

            input_ids_1 = emotion_batch['input_ids'].to(device)
            attention_mask_1 = emotion_batch['attention_mask'].to(device)
            labels_1 = emotion_batch['labels'].to(device)

            emotion_logits = model(input_ids_1,attention_mask_1)

            loss_1 = emotion_criterion(emotion_logits, labels_1.unsqueeze(1).float())



            e_loss += loss_1.item()

            loss_1.backward()
            nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.0,norm_type=2)
            optimizer.step()

        print(f"e_loss:{e_loss}")

        model.eval()
        with torch.no_grad():

            emo_pred = []
            emo_label = []

            for batch in tqdm(emotion_val_dataloader, desc="Emo_Eval"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                logits = model(input_ids, attention_mask)
                sigmoid = nn.Sigmoid()
                probs = sigmoid(logits)
                predictions = (probs > 0.5).to(torch.int8)

                emo_pred.extend(predictions.cpu().numpy().flatten())
                emo_label.extend(labels.cpu().numpy().flatten())
                # print("type:",type(references),type(predictions))# Tensor Tensor

            emo_result = {
                'accuracy': accuracy_score(emo_label,emo_pred),
                'precision': precision_score(emo_label,emo_pred,average='macro'),
                'recall': recall_score(emo_label,emo_pred,average='macro'),
                'f1': f1_score(emo_label,emo_pred,average='macro'),
            }
            print(emo_result)

        if max_accuracy < emo_result['accuracy']:
            max_accuracy = emo_result['accuracy']



        total_accuracy = emo_result['accuracy']
        early_stopping(total_accuracy, epoch, optimizer)

        if early_stopping.early_stop:
            print("早停！训练停止。")
            break

    K_emo_accuracy.append(max_accuracy)

print(f"{K_emo_accuracy},平均值：{np.mean(K_emo_accuracy)}")
