import argparse
import random
from functools import partial

import numpy as np
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.utils.data import DataLoader

from tqdm import tqdm

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch.optim as optim
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from SingleTaskDataLoad.glove_dataloader import *
from SingleTaskModels.model1_1 import SingletaskModel

from myutils.EarlyStop import EarlyStopping
from myutils.label_process import multilabel_process

from myutils.calculate_per_dim_accuracy import calculate_per_dim_accuracy, calculate_mbti_accuracy
from myutils.utils_loss import SmoothBCELoss
from SingleTaskDataLoad.Glove_loader import *


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

# 16 8
# 1e-5 5e-6
parser.add_argument("--checkpoint_path",type=str,default="/hy-tmp/check_points/essays_isear_checkpoint.pth",help="检查点保存路径")
parser.add_argument("--model_name",type=str,default="D:/models/bert-base-uncased",help="分词和训练的模型")
parser.add_argument("--batch_size",type=int,default=16,help="批量大小")
parser.add_argument("--k_fold",type=int,default=5,help="交叉验证折数")
parser.add_argument("--random_seed",type=int,default=67373,help="随机种子")
parser.add_argument("--epoch",type=int,default=30,help="训练次数")
parser.add_argument("--learning_rate1",type=float,default=2e-5,help="学习率1")
parser.add_argument("--learning_rate2",type=float,default=2e-6,help="学习率2")
parser.add_argument("--eps",type=float,default=1e-8,help="数值稳定参数")
parser.add_argument("--patience",type=int,default=5,help="早停忍耐度")
parser.add_argument("--personality_url",type=str,default="D:/datasets_1/Essays/原始数据/essays_revise.csv",help="人格数据集路径")
parser.add_argument("--Senticnet8_url",type=str,default="D:/datasets_1/affectivespace/affective.csv",help="情感数据集路径")
parser.add_argument("--pnd_max_length",type=int,default=1024,help="人格数据截断长度")
parser.add_argument("--use_senticnet_feature",type=bool,default=True,help="是否使用SenticNet提取额外的100维特征")
parser.add_argument("--step_size",type=int,default=0.1,help="warmup比例")
parser.add_argument("--gamma",type=float,default=0.7,help="学习率衰减倍数")

args = parser.parse_args()

set_all_seeds(args.random_seed)

tokenizers = AutoTokenizer.from_pretrained(args.model_name)

# 损失函数定义，使用标签平滑
criterion = nn.BCEWithLogitsLoss().to(device)

# 加载数据
p_texts, p_labels = read_data(args.personality_url)

skf = StratifiedKFold(n_splits=args.k_fold,shuffle=True,random_state=args.random_seed)

pnd_K = skf.split(p_texts,multilabel_process(p_labels))


K_per_metrics= {
    'EXT_acc':[],'NEU_acc':[],'AGR_acc':[],'CON_acc':[],'OPN_acc':[],
    'accuracy': [],
    'EXT_f1':[],'NEU_f1':[],'AGR_f1':[],'CON_f1':[],'OPN_f1':[],
    "macro-f1":[]
}

word_to_vec = load_GloVe("D:/Glove/glove.6B.300d.txt")
vocab, embedding_matrix = build_vocab_and_embedding_matrix(
    texts=p_texts,
    word_to_vec=word_to_vec,
    embedding_dim=300,
    min_freq=1
)


for fold, (p_train_index, p_test_index) in enumerate(pnd_K):
    print("*" * 50)
    print(f"第{fold + 1}折")
    print("*" * 50)
    # 加载dataloader
    train_dataset = TextDataset(p_texts[p_train_index],p_labels[p_train_index])
    test_dataset = TextDataset(p_texts[p_test_index],p_labels[p_test_index])
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=partial(collate_fn,vocab=vocab,MAX_SEQUENCE_LENGTH=args.pnd_max_length),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=partial(collate_fn,vocab=vocab,MAX_SEQUENCE_LENGTH=args.pnd_max_length),
    )
    # 加载嵌入模型
    model = SingletaskModel(embedding_matrix).to(device)
    # 计算参数
    parameters = list(model.parameters())
    print("训练参数量：", sum(p.numel() for p in parameters if p.requires_grad))


    # 定义分层学习率
    optimizer = optim.AdamW([
        # {'params': model.bert.parameters(), 'lr': args.learning_rate1},
        {'params': model.personality_head.parameters(), 'lr': args.learning_rate2},
        {'params': model.mamba.parameters(), 'lr': args.learning_rate2},
    ])
    print("mamba参数:",sum(p.numel() for p in list(model.mamba.parameters()) if p.requires_grad))
    # 学习率衰减
    num_training_steps = len(train_loader) * args.epoch
    num_warmup_steps = int(num_training_steps * args.step_size)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    # 早停策略
    early_stopping = EarlyStopping(patience=args.patience, verbose=True, path=args.checkpoint_path)


    p_max_acc, p_f1 = 0.0, 0.0
    EXT, NEU, AGR, CON, OPN = 0.0, 0.0, 0.0, 0.0, 0.0
    EXT_f1, NEU_f1, AGR_f1, CON_f1, OPN_f1 = 0.0, 0.0, 0.0, 0.0, 0.0
    for epoch in range(args.epoch):

        print(f"Epoch:{epoch + 1}")
        # 训练阶段
        p_train_loss = 0.0
        model.train()
        print("----- 训练阶段 -----")
        for input_ids, lengths, labels in tqdm(train_loader, desc="Train..."):

            optimizer.zero_grad()
            output = model(input_ids.to(device), lengths.to(device))
            loss = criterion(output, labels.to(device))

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.0,norm_type=2)

            optimizer.step()
            scheduler.step()

        # 验证阶段
        model.eval()
        print("----- 验证阶段 -----")
        with torch.no_grad():
            pnd_pred, pnd_label = [], []

            for input_ids, lengths, labels in tqdm(test_loader, desc="Eval"):

                output = model(input_ids.to(device), lengths.to(device))

                sigmoid = nn.Sigmoid()
                probs = sigmoid(output)
                predictions = (probs > 0.5).to(torch.int8)
                pnd_pred.extend(predictions.cpu().numpy())
                pnd_label.extend(labels.cpu().numpy())

        ext, neu, agr, con, opn = calculate_per_dim_accuracy(pnd_label, pnd_pred)
        f1_scores_per_dim = f1_score(pnd_label, pnd_pred, average=None, zero_division=0)

        per_result = {
            'ext_acc': ext,
            'neu_acc': neu,
            'agr_acc': agr,
            'con_acc': con,
            'opn_acc': opn,
            'accuracy': accuracy_score(np.concatenate(pnd_label), np.concatenate(pnd_pred)),
            'EXT_f1': f1_scores_per_dim[0],
            'NEU_f1': f1_scores_per_dim[1],
            'AGR_f1': f1_scores_per_dim[2],
            'CON_f1': f1_scores_per_dim[3],
            'OPN_f1': f1_scores_per_dim[4],
            'macro-f1': f1_score(pnd_label, pnd_pred, average='macro'),
        }
        print("----- 模型验证结果 -----")
        formatted_result = {k: f"{v:.4f}" for k, v in per_result.items()}
        print(formatted_result)
        total_accuracy = per_result['accuracy']


        early_stopping(total_accuracy, epoch, checkpoint=None)

        if early_stopping.early_stop:
            print("早停！训练停止。")
            break



        if early_stopping.update == True:

            EXT = per_result['ext_acc']
            NEU = per_result['neu_acc']
            AGR = per_result['agr_acc']
            CON = per_result['con_acc']
            OPN = per_result['opn_acc']
            p_max_acc = per_result['accuracy']
            EXT_f1 = per_result['EXT_f1']
            NEU_f1 = per_result['NEU_f1']
            AGR_f1 = per_result['AGR_f1']
            CON_f1 = per_result['CON_f1']
            OPN_f1 = per_result['OPN_f1']
            p_f1 = per_result['macro-f1']

        if early_stopping.early_stop:
            print("早停！训练停止。")
            break


    K_per_metrics['EXT_acc'].append(EXT)
    K_per_metrics['NEU_acc'].append(NEU)
    K_per_metrics['AGR_acc'].append(AGR)
    K_per_metrics['CON_acc'].append(CON)
    K_per_metrics['OPN_acc'].append(OPN)
    K_per_metrics['accuracy'].append(p_max_acc)
    K_per_metrics['EXT_f1'].append(EXT_f1)
    K_per_metrics['NEU_f1'].append(NEU_f1)
    K_per_metrics['AGR_f1'].append(AGR_f1)
    K_per_metrics['CON_f1'].append(CON_f1)
    K_per_metrics['OPN_f1'].append(OPN_f1)
    K_per_metrics['macro-f1'].append(p_f1)


# total_metrics = {
#     'pnd_acc': K_per_metrics['accuracy'],
#     'pnd_f1': K_per_metrics['f1'],
# }
# df = pd.DataFrame(total_metrics)
# print(df)

print("personality metrics:")
for key, value in K_per_metrics.items():
    print(f"{key}:{value}")
    print(f"{key}: {np.mean(value):.4f}")