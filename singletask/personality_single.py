import argparse

import numpy as np
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.cuda.amp import GradScaler,autocast

from tqdm import tqdm

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch.optim as optim
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from data_loader.load_per_isear_ori import *
from SingleTaskModels.model1 import SingletaskModel
from myutils.EarlyStop import EarlyStopping
from myutils.label_process import multilabel_process

from myutils.calculate_per_dim_accuracy import calculate_per_dim_accuracy, calculate_mbti_accuracy
from myutils.utils_loss import SmoothBCELoss


def set_all_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"All random seeds set to {seed}")

set_all_seeds(67373)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

parser = argparse.ArgumentParser()

parser.add_argument("--checkpoint_path",type=str,default="/hy-tmp/check_points/essays_isear_checkpoint.pth",help="检查点保存路径")
parser.add_argument("--model_url",type=str,default="/hy-tmp/models/bert-base-uncased",help="分词和训练的模型")
parser.add_argument("--batch_size",type=int,default=16,help="批量大小")
parser.add_argument("--k_fold",type=int,default=5,help="交叉验证折数")
parser.add_argument("--personality_DataName",type=str,default="mbti",help="人格任务的数据集名称")
parser.add_argument("--random_seed",type=int,default=67373,help="随机种子")
parser.add_argument("--epoch",type=int,default=20,help="训练次数")
parser.add_argument("--learning_rate1",type=float,default=1e-5,help="学习率1")
parser.add_argument("--learning_rate2",type=float,default=1e-4,help="学习率2")
parser.add_argument("--eps",type=float,default=1e-8,help="数值稳定参数")
parser.add_argument("--patience",type=int,default=5,help="早停忍耐度")
parser.add_argument("--personality_url",type=str,default="/hy-tmp/dataset/MBTI/原始数据/少数类1.csv",help="人格数据集路径")
parser.add_argument("--Senticnet8_url",type=str,default="/hy-tmp/dataset/affectivespace/affective.csv",help="情感数据集路径")
parser.add_argument("--pnd_max_length",type=int,default=512,help="人格数据截断长度")
parser.add_argument("--use_senticnet_feature",type=bool,default=True,help="是否使用SenticNet提取额外的100维特征")
parser.add_argument("--step_size",type=int,default=0.1,help="warmup比例")
parser.add_argument("--gamma",type=float,default=0.1,help="学习率衰减倍数")

args = parser.parse_args()

set_all_seeds(args.random_seed)

tokenizers = AutoTokenizer.from_pretrained(args.model_url)

# 损失函数定义，使用标签平滑
pnd_criterion = SmoothBCELoss(0.1).to(device)

# 加载数据
p_texts, p_labels = read_data(args.personality_url,args.personality_DataName)

skf = StratifiedKFold(n_splits=args.k_fold,shuffle=True,random_state=args.random_seed)

pnd_K = skf.split(p_texts,multilabel_process(p_labels))

K_per_metrics= {
    'IE_acc':[],
    'NS_acc':[],
    'TF_acc':[],
    'PJ_acc':[],
    'accuracy':[],
    'IE_f1': [],
    'NS_f1':[],
    'TF_f1':[],
    'PJ_f1':[],
    'macro-f1':[]
}
for fold, (p_train_index, p_test_index) in enumerate(pnd_K):
    print("*" * 50)
    print(f"第{fold + 1}折")
    print("*" * 50)
    # 加载dataloader
    pnd_train_dataloader,pnd_val_dataloader = create_dataloaders(
        batch_size=args.batch_size,
        text=p_texts,
        labels=p_labels,
        tokenizer=tokenizers,
        max_length=args.pnd_max_length,
        train_idx=p_train_index,
        val_idx=p_test_index,
        seed=args.random_seed,
    )
    # 加载嵌入模型
    model = SingletaskModel().to(device)
    # 计算参数
    parameters = list(model.parameters())
    print("训练参数量：", sum(p.numel() for p in parameters if p.requires_grad))

    # 定义分层学习率
    optimizer = optim.AdamW([
        {'params': model.bert.parameters(), 'lr': args.learning_rate1},
        {'params': model.personality_head.parameters(), 'lr': args.learning_rate2},
    ])

    # 学习率衰减
    num_training_steps = len(pnd_train_dataloader) * args.epoch
    num_warmup_steps = int(num_training_steps * args.step_size)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    # 早停策略
    early_stopping = EarlyStopping(patience=args.patience, verbose=True, path=args.checkpoint_path)


    p_max_acc, p_f1 = 0.0, 0.0
    ie, ns, tf, pj = 0.0, 0.0, 0.0, 0.0
    f1,f2,f3,f4 = 0.0, 0.0, 0.0, 0.0
    for epoch in range(args.epoch):

        print(f"Epoch:{epoch + 1}")
        # 训练阶段
        p_train_loss = 0.0
        model.train()
        print("----- 训练阶段 -----")
        for batch in tqdm(pnd_train_dataloader, desc="Train..."):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device).float()

            output = model(input_ids, attention_mask)
            loss = pnd_criterion(output, labels)
            p_train_loss += loss.item()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2)
            optimizer.step()
            scheduler.step()


        print(f"损失情况： 人格任务损失:{p_train_loss}")

        # 验证阶段
        model.eval()
        print("----- 验证阶段 -----")
        with torch.no_grad():
            pnd_pred, pnd_label = [], []

            for batch in tqdm(pnd_val_dataloader, desc="Eval"):

                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                output = model(input_ids, attention_mask)
                probs = torch.sigmoid(output)
                predictions = (probs > 0.5).to(torch.long)
                pnd_pred.extend(predictions.cpu().numpy())
                pnd_label.extend(labels.cpu().numpy())

        I_E,N_S,T_F,P_J = calculate_mbti_accuracy(pnd_label, pnd_pred)
        f1_scores_per_dim = f1_score(pnd_label, pnd_pred, average=None, zero_division=0)
        per_result = {
            'I/E_acc': I_E,
            'N/S_acc': N_S,
            'T/F_acc': T_F,
            'P/J_acc': P_J,
            'accuracy': accuracy_score(np.concatenate(pnd_label), np.concatenate(pnd_pred)),
            'I/E_f1': f1_scores_per_dim[0],
            'N/S_f1': f1_scores_per_dim[1],
            'T/F_f1': f1_scores_per_dim[2],
            'P/J_f1': f1_scores_per_dim[3],
            'macro-f1':f1_score(pnd_label, pnd_pred, average='macro'),
        }

        print("----- 模型验证结果 -----")
        formatted_result = {k: v for k, v in per_result.items()}
        print(formatted_result)
        total_accuracy = per_result['accuracy']

        early_stopping(total_accuracy, epoch, checkpoint=None)

        if early_stopping.early_stop:
            print("早停！训练停止。")
            break



        if early_stopping.update == True:
            p_max_acc = per_result['accuracy']
            ie = per_result['I/E_acc']
            ns = per_result['N/S_acc']
            tf = per_result['T/F_acc']
            pj = per_result['P/J_acc']
            f1 = per_result['I/E_f1']
            f2 = per_result['N/S_f1']
            f3 = per_result['T/F_f1']
            f4 = per_result['P/J_f1']
            p_f1 = per_result['macro-f1']

        if early_stopping.early_stop:
            print("早停！训练停止。")
            break
    K_per_metrics['IE_acc'].append(ie)
    K_per_metrics['NS_acc'].append(ns)
    K_per_metrics['TF_acc'].append(tf)
    K_per_metrics['PJ_acc'].append(pj)
    K_per_metrics['IE_f1'].append(f1)
    K_per_metrics['NS_f1'].append(f2)
    K_per_metrics['TF_f1'].append(f3)
    K_per_metrics['PJ_f1'].append(f4)
    K_per_metrics['accuracy'].append(p_max_acc)
    K_per_metrics['macro-f1'].append(p_f1)

print("personality metrics:")
for key, value in K_per_metrics.items():
    print(f"{key}:{value}")
    print(f"{key}: {np.mean(value):.4f}")