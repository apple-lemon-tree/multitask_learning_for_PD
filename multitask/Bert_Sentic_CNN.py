import argparse
import os
import random

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch import nn
from tqdm import tqdm

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from transformers import AutoTokenizer

from data_loader.bert_cnn_dataloader import *
from MultitaskModels.model4 import *
from myutils.EarlyStop import EarlyStopping
from myutils.label_process import multilabel_process
from myutils.AffectiveSpaceFeatureExtractor import AffectiveFeatureExtractor
from myutils.weight_methods import MGDA,NashMTL,IMTLG
from myutils.calculate_per_dim_accuracy import calculate_per_dim_accuracy

def set_all_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"All random seeds set to {seed}")

set_all_seeds(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

pnd_class_freq = [
    2947, 2602,3688, 3190, 5159
]

parser = argparse.ArgumentParser()

parser.add_argument("--checkpoint_path",type=str,default="../check_points/combinedFeature_essays_isear_checkpoint.pth",help="检查点保存路径")
parser.add_argument("--model_url",type=str,default="/hy-tmp/models/bert-base-uncased",help="分词和训练的模型")
parser.add_argument("--batch_size",type=int,default=16,help="批量大小")
parser.add_argument("--k_fold",type=int,default=5,help="交叉验证折数")
parser.add_argument("--personality_DataName",type=str,default="myPersonality",help="人格任务的数据集名称")
parser.add_argument("--emotion_DataName",type=str,default="isear",help="情感任务的数据集名称")
parser.add_argument("--random_seed",type=int,default=42,help="随机种子")
parser.add_argument("--epoch",type=int,default=50,help="训练次数")
parser.add_argument("--learning_rate",type=float,default=1e-5,help="学习率")
parser.add_argument("--weight_decay",type=float,default=0.01,help="权重衰减")
parser.add_argument("--eps",type=float,default=1e-8,help="数值稳定参数")
parser.add_argument("--patience",type=int,default=5,help="早停忍耐度")
parser.add_argument("--personality_url",type=str,default="/hy-tmp/dataset/myPersonality/原始数据/Personality_1.csv",help="人格数据集路径")
parser.add_argument("--emotion_url",type=str,default="/hy-tmp/dataset/ISEAR/原始数据/ISEAR_1.csv",help="情感数据集路径")
parser.add_argument("--Senticnet8_url",type=str,default="/hy-tmp/dataset/affectivespace/affectivespace.xlsx",help="情感数据集路径")
parser.add_argument("--pnd_max_length",type=int,default=128,help="人格数据截断长度")
parser.add_argument("--emo_max_length",type=int,default=128,help="情感数据截断长度")
parser.add_argument("--use_senticnet_feature",type=bool,default=True,help="是否使用SenticNet提取额外的100维特征")
parser.add_argument("--step_size",type=int,default=10,help="学习率衰减频率")
parser.add_argument("--gamma",type=float,default=0.1,help="学习率衰减倍数")

args = parser.parse_args()

set_all_seeds(args.random_seed)

tokenizers = AutoTokenizer.from_pretrained(args.model_url)

k_fold = args.k_fold

p_text, p_labels = read_data(args.personality_url,args.personality_DataName)
e_text, e_labels = read_data(args.emotion_url,args.emotion_DataName)

skf = StratifiedKFold(n_splits=args.k_fold, shuffle=True,random_state=args.random_seed)

personality_criterion = nn.BCEWithLogitsLoss().to(device)
# pnd_criterion = ResampleLoss(reweight_func='rebalance', loss_weight=1.0,
#                              focal=dict(focal=True, alpha=0.5, gamma=2),
#                              logit_reg=dict(init_bias=0.01, neg_scale=1.0),
#                              map_param=dict(alpha=0.5, beta=10, gamma=0.3),
#                              class_freq=pnd_class_freq, train_num=6941)
emotion_criterion = nn.CrossEntropyLoss().to(device)
weighting_method = NashMTL(n_tasks=2,device=device)

myPersonality_K = skf.split(p_text, multilabel_process(p_labels))

isear_K = skf.split(e_text, e_labels)

K_per_metrics= {'EXT_acc':[],'NEU_acc':[],'AGR_acc':[],'CON_acc':[],'OPE_acc':[],'pnd_Accuracy': [],'pnd_macroF1': []}
K_emo_metrics = {'emo_Accuracy': [],'emo_macroF1': []}

extractor = AffectiveFeatureExtractor(args.Senticnet8_url) if args.use_senticnet_feature else None

for fold, ((p_train_index, p_test_index),(e_train_index, e_test_index)) in enumerate(zip(myPersonality_K, isear_K)):
    print("*"*50)
    print(f"第{fold + 1}折")
    print("*" * 50)

    Embedding_model = GetEmbeddings(args.model_url,extractor).to(device)
    Feature_process_model = HMD_GCF_Module().to(device)
    Classification_model = ClassficationHead(args.personality_DataName,args.emotion_DataName).to(device)


    parameters = list(Embedding_model.parameters()) + list(Feature_process_model.parameters()) + list(Classification_model.parameters())
    total_params = sum(p.numel() for p in parameters if p.requires_grad)
    print("模型可训练参数量：", total_params)

    optimizer = optim.AdamW(parameters, lr=args.learning_rate, weight_decay=args.weight_decay, eps=args.eps)
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    early_stopping = EarlyStopping(patience=args.patience, verbose=True,path=args.checkpoint_path)

    personality_train_dataloader,personality_val_dataloader = (
        create_dataloaders(
            tokenizer=tokenizers,batch_size=args.batch_size,max_length=args.pnd_max_length,
            text=p_text,labels=p_labels,train_idx=p_train_index,val_idx=p_test_index,seed=args.random_seed,
        )
    )
    emotion_train_dataloader, emotion_val_dataloader = (
        create_dataloaders(
            tokenizer=tokenizers,batch_size=args.batch_size,max_length=args.emo_max_length,
            text=e_text,labels=e_labels,train_idx=e_train_index, val_idx=e_test_index,seed=args.random_seed,
        )
    )

    per_iter = iter(personality_train_dataloader)
    emo_iter = iter(emotion_train_dataloader)

    p_max_acc, e_max_acc, p_f1, e_f1 = 0.0, 0.0, 0.0, 0.0
    ext, neu, agr, con, ope = 0.0, 0.0, 0.0, 0.0, 0.0

    for epoch in range(args.epoch):
        print(f"Epoch:{epoch + 1}")

        torch.cuda.empty_cache()
        Embedding_model.train()
        Feature_process_model.train()
        Classification_model.train()

        batchs_num = max(len(personality_train_dataloader),len(emotion_train_dataloader))

        p_loss, i_loss = 0.0, 0.0
        for i in tqdm(range(batchs_num),desc="Train..."):
            optimizer.zero_grad()
            try:
                personality_batch = next(per_iter)
            except StopIteration:
                per_iter = iter(personality_train_dataloader)
                personality_batch = next(per_iter)

            try:
                emotion_batch = next(emo_iter)
            except StopIteration:
                emo_iter = iter(emotion_train_dataloader)
                emotion_batch = next(emo_iter)

            texts_1 = personality_batch['texts']
            input_ids_1 = personality_batch['input_ids'].to(device)
            attention_mask_1 = personality_batch['attention_mask'].to(device)
            labels_1 = personality_batch['labels'].to(device).float()

            texts_2 = emotion_batch['texts']
            input_ids_2 = emotion_batch['input_ids'].to(device)
            attention_mask_2 = emotion_batch['attention_mask'].to(device)
            labels_2 = emotion_batch['labels'].to(device)

            # affective_features_1 = torch.from_numpy(extractor.extract_features_batch(texts_1)).to(device)
            # affective_features_2 = torch.from_numpy(extractor.extract_features_batch(texts_2)).to(device)

            semantic_embeddings_1, senticnet_embeddings_1 = Embedding_model(input_ids_1, attention_mask_1,texts_1)
            semantic_embeddings_2, senticnet_embeddings_2 = Embedding_model(input_ids_2, attention_mask_2, texts_2)

            personality_embedding= Feature_process_model(
                semantic_embeddings_1,
                senticnet_embeddings_1,
            )
            personality_output = Classification_model(personality_embedding,args.personality_DataName)
            loss_1 = personality_criterion(personality_output, labels_1)

            emotion_embedding = Feature_process_model(
                semantic_embeddings_2,
                senticnet_embeddings_2,
            )
            emotion_output = Classification_model(emotion_embedding,args.emotion_DataName)
            loss_2 = emotion_criterion(emotion_output, labels_2)

            losses = torch.stack(
                (
                    loss_1,
                    loss_2
                )
            )
            weighted_loss, extra_outputs = weighting_method.backward(
                losses=losses,
                shared_parameters=list(Embedding_model.parameters()) + list(Feature_process_model.parameters())
            )
            # print(f"任务权重: {extra_outputs['weights']}")
            # loss = loss_1 + loss_2

            p_loss += loss_1.item()
            i_loss += loss_2.item()

            # loss.backward()
            nn.utils.clip_grad_norm_(parameters,max_norm=1.0,norm_type=2)
            optimizer.step()

        print(f"p_loss:{p_loss},i_loss:{i_loss}")

        Embedding_model.eval()
        Feature_process_model.eval()
        Classification_model.eval()
        with torch.no_grad():

            pnd_pred = []
            pnd_label = []
            emo_pred = []
            emo_label = []

            for batch in tqdm(personality_val_dataloader, desc="PND_Eval"):
                texts = batch['texts']
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                semantic_embeddings, senticnet_embeddings = Embedding_model(input_ids, attention_mask,texts)
                embedding = Feature_process_model(semantic_embeddings,senticnet_embeddings)
                output = Classification_model(embedding,args.personality_DataName)

                predictions = (output > 0.5).to(torch.int8)

                pnd_pred.extend(predictions.cpu().numpy())
                pnd_label.extend(labels.cpu().numpy())
                # print("type:",type(references),type(predictions))# Tensor Tensor


            for batch in tqdm(emotion_val_dataloader, desc="Emo_Eval"):
                texts = batch['texts']
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                semantic_embeddings, senticnet_embeddings = Embedding_model(input_ids, attention_mask, texts)
                embedding = Feature_process_model(semantic_embeddings, senticnet_embeddings)
                output = Classification_model(embedding,args.emotion_DataName)

                probs = torch.argmax(output,dim=1)

                emo_pred.extend(probs.cpu().numpy())
                emo_label.extend(labels.cpu().numpy())

            ext_acc, neu_acc, agr_acc, con_acc, ope_acc = calculate_per_dim_accuracy(pnd_label, pnd_pred)
            per_result = {
                'EXT_acc': ext_acc,
                'NEU_acc': neu_acc,
                'AGR_acc': agr_acc,
                'CON_acc': con_acc,
                'OPE_acc': ope_acc,
                'accuracy': accuracy_score(np.concatenate(pnd_label), np.concatenate(pnd_pred)),
                'precision': precision_score(pnd_label, pnd_pred, average='macro'),
                'recall': recall_score(pnd_label, pnd_pred, average='macro'),
                'macro_f1': f1_score(pnd_label, pnd_pred, average='macro'),
            }

            emo_result = {
                'accuracy': accuracy_score(emo_label, emo_pred),
                'precision': precision_score(emo_label, emo_pred, average='macro'),
                'recall': recall_score(emo_label, emo_pred, average='macro'),
                'macro_f1': f1_score(emo_label, emo_pred, average='macro'),
            }
            print(per_result)
            print(emo_result)
            total_accuracy = per_result['accuracy'] + emo_result['accuracy']
            early_stopping(total_accuracy, epoch, optimizer)

        if early_stopping.update == True:
            p_max_acc = per_result['accuracy']
            ext = per_result['EXT_acc']
            neu = per_result['NEU_acc']
            agr = per_result['AGR_acc']
            con = per_result['CON_acc']
            ope = per_result['OPE_acc']
            p_f1 = per_result['macro_f1']
            e_max_acc = emo_result['accuracy']
            e_f1 = emo_result['macro_f1']
        if early_stopping.early_stop:
            print("早停！训练停止。")
            break

        scheduler.step()

    K_per_metrics['pnd_Accuracy'].append(p_max_acc)

    K_per_metrics['EXT_acc'].append(ext)
    K_per_metrics['NEU_acc'].append(neu)
    K_per_metrics['AGR_acc'].append(agr)
    K_per_metrics['CON_acc'].append(con)
    K_per_metrics['OPE_acc'].append(ope)

    K_per_metrics['pnd_macroF1'].append(p_f1)

    K_emo_metrics['emo_Accuracy'].append(e_max_acc)
    K_emo_metrics['emo_macroF1'].append(e_f1)

total_metrics = {
    'pnd_acc': K_per_metrics['accuracy'],
    'pnd_f1': K_per_metrics['f1'],
    'emo_acc': K_emo_metrics['accuracy'],
    'emo_f1': K_emo_metrics['f1'],
}
df = pd.DataFrame(total_metrics)
print(df)

print("personality metrics:")
for key, value in K_per_metrics.items():
    print(f"{key}:{value}")
    print(f"{key}: {np.mean(value)}")
print("emotion metrics:")
for key, value in K_emo_metrics.items():
    print(f"{key}:{value}")
    print(f"{key}: {np.mean(value)}")