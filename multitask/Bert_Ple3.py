import argparse

import numpy as np
from sklearn.model_selection import StratifiedKFold

from tqdm import tqdm

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR,CosineAnnealingLR
from transformers import AutoTokenizer

from data_loader.data_sentic_load3 import *
from MultitaskModels.model3 import *
from myutils.EarlyStop import EarlyStopping
from myutils.label_process import multilabel_process
from myutils.Token_SenticNet import AffectiveFeatureExtractor
from myutils.weight_methods import MGDA,NashMTL,IMTLG,Uncertainty
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

pnd_class_freq = [
    2947, 2602,3688, 3190, 5159
]

parser = argparse.ArgumentParser()

parser.add_argument("--checkpoint_path",type=str,default="/hy-tmp/check_points/essays_isear_checkpoint.pth",help="检查点保存路径")
parser.add_argument("--model_url",type=str,default="/hy-tmp/models/bert-base-uncased",help="分词和训练的模型")
parser.add_argument("--batch_size",type=int,default=8,help="批量大小")
parser.add_argument("--k_fold",type=int,default=5,help="交叉验证折数")
parser.add_argument("--personality_DataName",type=str,default="essays",help="人格任务的数据集名称")
parser.add_argument("--emotion_DataName",type=str,default="isear",help="情感任务的数据集名称")
parser.add_argument("--random_seed",type=int,default=67373,help="随机种子")
parser.add_argument("--epoch",type=int,default=50,help="训练次数")
parser.add_argument("--learning_rate1",type=float,default=5e-6,help="学习率1")
parser.add_argument("--learning_rate2",type=float,default=1e-5,help="学习率2")
parser.add_argument("--eps",type=float,default=1e-8,help="数值稳定参数")
parser.add_argument("--patience",type=int,default=5,help="早停忍耐度")
parser.add_argument("--personality_url",type=str,default="/hy-tmp/dataset/Essays/原始数据/essays_revise.csv",help="人格数据集路径")
parser.add_argument("--emotion_url",type=str,default="/hy-tmp/dataset/ISEAR/原始数据/ISEAR_1.csv",help="情感数据集路径")
parser.add_argument("--Senticnet8_url",type=str,default="/hy-tmp/dataset/affectivespace/affective.csv",help="情感数据集路径")
parser.add_argument("--pnd_max_length",type=int,default=256,help="人格数据截断长度")
parser.add_argument("--emo_max_length",type=int,default=256,help="情感数据截断长度")
parser.add_argument("--use_senticnet_feature",type=bool,default=True,help="是否使用SenticNet提取额外的100维特征")
parser.add_argument("--step_size",type=int,default=10,help="学习率衰减频率")
parser.add_argument("--gamma",type=float,default=0.3,help="学习率衰减倍数")

args = parser.parse_args()

set_all_seeds(args.random_seed)

tokenizers = AutoTokenizer.from_pretrained(args.model_url)

# 损失函数定义，使用标签平滑
pnd_criterion = SmoothBCELoss(0.1).to(device)
emo_criterion = nn.CrossEntropyLoss(label_smoothing=0.1).to(device)
# emotion_criterion = SmoothBCELoss(0.1).to(device)

# 损失加权定义
logsigma = torch.tensor([0.0] * 2, device=device, requires_grad=True)
weighting_method = NashMTL(n_tasks=2,device=device)

# 情感特征提取器
extractor = AffectiveFeatureExtractor(args.Senticnet8_url)

# 加载数据
p_texts, p_labels = read_data(args.personality_url,args.personality_DataName)
e_texts, e_labels = read_data(args.emotion_url,args.emotion_DataName)

skf = StratifiedKFold(n_splits=args.k_fold,shuffle=True,random_state=args.random_seed)

pnd_K = skf.split(p_texts,multilabel_process(p_labels))
emo_K = skf.split(e_texts,e_labels)


K_per_metrics= {'EXT_acc':[],'NEU_acc':[],'AGR_acc':[],'CON_acc':[],'OPN_acc':[],'accuracy': [],'f1': []}
# K_per_metrics= {'IE_acc':[],'NS_acc':[],'TF_acc':[],'PJ_acc':[],'accuracy': [],'f1': []}
K_emo_metrics = {'accuracy': [],'f1': []}
for fold, ((p_train_index, p_test_index),(e_train_index, e_test_index)) in enumerate(zip(pnd_K, emo_K)):
    print("*" * 50)
    print(f"第{fold + 1}折")
    print("*" * 50)
    # 加载dataloader
    pnd_train_dataloader,pnd_val_dataloader = create_dataloaders(
        batch_size=args.batch_size,
        texts=p_texts,
        labels=p_labels,
        tokenizer=tokenizers,
        max_length=args.pnd_max_length,
        train_idx=p_train_index,
        val_idx=p_test_index,
        extractor=extractor,
        seed=args.random_seed,
    )
    emo_train_dataloader,emo_val_dataloader = create_dataloaders(
        batch_size=args.batch_size,
        texts=e_texts,
        labels=e_labels,
        tokenizer=tokenizers,
        max_length=args.emo_max_length,
        train_idx=e_train_index,
        val_idx=e_test_index,
        extractor=extractor,
        seed=args.random_seed,
    )
    # 加载嵌入模型
    Embedding_model = TextEmbedding(args.model_url,args.use_senticnet_feature).to(device)
    # 加载主干模型
    # model = PLE(
    #     input_dim=768,
    #     expert_dim=768,
    # ).to(device)
    model = MultitaskModel().to(device)
    # 计算参数
    parameters = list(model.parameters()) + list(Embedding_model.parameters())
    print("训练参数量：", sum(p.numel() for p in parameters if p.requires_grad))

    # 定义分层学习率
    optimizer = optim.AdamW([
        {'params': Embedding_model.parameters(), 'lr': args.learning_rate1},
        {'params': model.parameters(), 'lr': args.learning_rate2},
    ])
    # 学习率衰减
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    # 早停策略
    early_stopping = EarlyStopping(patience=args.patience, verbose=True, path=args.checkpoint_path)



    train_MAX,train_MIN = max(len(pnd_train_dataloader),len(emo_train_dataloader)),min(len(pnd_train_dataloader),len(emo_train_dataloader))
    val_MAX,val_MIN = max(len(pnd_val_dataloader),len(emo_val_dataloader)),min(len(pnd_val_dataloader),len(emo_val_dataloader))

    p_max_acc, e_max_acc, p_f1, e_f1 = 0.0, 0.0, 0.0, 0.0
    EXT, NEU, AGR, CON, OPN = 0.0, 0.0, 0.0, 0.0, 0.0
    ie, ns, tf, pj = 0.0, 0.0, 0.0, 0.0
    for epoch in range(args.epoch):
        pnd_iter = iter(pnd_train_dataloader)
        emo_iter = iter(emo_train_dataloader)
        pnd_val_iter = iter(pnd_val_dataloader)
        emo_val_iter = iter(emo_val_dataloader)
        print(f"Epoch:{epoch + 1}")
        # 训练阶段
        p_train_loss, e_train_loss = 0.0, 0.0
        Embedding_model.train()
        model.train()
        print("----- 训练阶段 -----")
        for i in tqdm(range(train_MIN), desc="Train..."):
            optimizer.zero_grad()
            pnd_batch = next(pnd_iter)
            emo_batch = next(emo_iter)

            input_ids1 = pnd_batch['input_ids'].to(device)
            attention_mask1 = pnd_batch['attention_mask'].to(device)
            labels1 = pnd_batch['labels'].to(device).float()
            sentic_fea1 = pnd_batch['sentic_fea'].to(device)

            input_ids2 = emo_batch['input_ids'].to(device)
            attention_mask2 = emo_batch['attention_mask'].to(device)
            labels2 = emo_batch['labels'].to(device)
            sentic_fea2 = emo_batch['sentic_fea'].to(device)

            pnd_embeddings = Embedding_model(input_ids1, attention_mask1, sentic_fea1)
            emo_embeddings = Embedding_model(input_ids2, attention_mask2, sentic_fea2)

            pnd_output, emo_output = model(pnd_embeddings, emo_embeddings)
            loss_1 = pnd_criterion(pnd_output, labels1)
            loss_2 = emo_criterion(emo_output, labels2)
            losses = torch.stack((loss_1, loss_2))
            # shared_parameters = list(Embedding_model.parameters()) + list(model.shared_params())
            # task_specific_params = list(model.task_specific_params())
            shared_parameters = list(Embedding_model.parameters())
            task_specific_params = list(model.parameters())
            weighted_loss, extra_outputs = weighting_method.backward(
                losses=losses,
                shared_parameters=shared_parameters,
                task_specific_params=task_specific_params,
            )
            # loss = loss_1 + loss_2
            p_train_loss += loss_1.item()
            e_train_loss += loss_2.item()

            # loss.backward()
            # nn.utils.clip_grad_norm_(Embedding_model.parameters(), max_norm=1.0, norm_type=2)
            # nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.0,norm_type=2)
            optimizer.step()

        print(f"损失情况： 人格任务损失:{p_train_loss},情感任务损失:{e_train_loss}")

        # 验证阶段
        Embedding_model.eval()
        model.eval()
        print("----- 验证阶段 -----")
        with torch.no_grad():
            pnd_pred, pnd_label, emo_pred, emo_label = [], [], [], []

            for i in tqdm(range(val_MIN), desc="Eval"):

                pnd_val_batch = next(pnd_val_iter)
                emo_val_batch = next(emo_val_iter)

                input_ids1 = pnd_val_batch['input_ids'].to(device)
                attention_mask1 = pnd_val_batch['attention_mask'].to(device)
                labels1 = pnd_val_batch['labels'].to(device)
                sentic_fea1 = pnd_val_batch['sentic_fea'].to(device)

                input_ids2 = emo_val_batch['input_ids'].to(device)
                attention_mask2 = emo_val_batch['attention_mask'].to(device)
                labels2 = emo_val_batch['labels'].to(device)
                sentic_fea2 = emo_val_batch['sentic_fea'].to(device)

                pnd_embeddings = Embedding_model(input_ids1, attention_mask1, sentic_fea1)
                emo_embeddings = Embedding_model(input_ids2, attention_mask2, sentic_fea2)
                out1, out2 = model(pnd_embeddings, emo_embeddings)
                sigmoid = nn.Sigmoid()
                probs = sigmoid(out1)
                predictions1 = (probs > 0.5).to(torch.int8)
                pnd_pred.extend(predictions1.cpu().numpy())
                pnd_label.extend(labels1.cpu().numpy())

                predictions2 = torch.argmax(out2, dim=1)
                emo_pred.extend(predictions2.cpu().numpy())
                emo_label.extend(labels2.cpu().numpy())

        ext, neu, agr, con, opn = calculate_per_dim_accuracy(pnd_label, pnd_pred)
        # I_E,N_S,T_F,P_J = calculate_mbti_accuracy(pnd_label, pnd_pred)
        # per_result = {
        #     'I/E_acc': I_E,
        #     'N/S_acc': N_S,
        #     'T/F_acc': T_F,
        #     'P/J_acc': P_J,
        #     'accuracy': accuracy_score(np.concatenate(pnd_label), np.concatenate(pnd_pred)),
        #     'precision': precision_score(pnd_label, pnd_pred, average='macro'),
        #     'recall': recall_score(pnd_label, pnd_pred, average='macro'),
        #     'f1': f1_score(pnd_label, pnd_pred, average='macro'),
        # }
        per_result = {
            'ext_acc': ext,
            'neu_acc': neu,
            'agr_acc': agr,
            'con_acc': con,
            'opn_acc': opn,
            'accuracy': accuracy_score(np.concatenate(pnd_label), np.concatenate(pnd_pred)),
            'precision': precision_score(pnd_label, pnd_pred, average='macro'),
            'recall': recall_score(pnd_label, pnd_pred, average='macro'),
            'f1': f1_score(pnd_label, pnd_pred, average='macro'),
        }

        emo_result = {
            'accuracy': accuracy_score(emo_label, emo_pred),
            'precision': precision_score(emo_label, emo_pred, average='macro'),
            'recall': recall_score(emo_label, emo_pred, average='macro'),
            'f1': f1_score(emo_label, emo_pred, average='macro'),
        }
        print("----- 模型验证结果 -----")
        print(per_result)
        print(emo_result)
        total_accuracy = 0.7 * per_result['accuracy'] + 0.3 * emo_result['accuracy']

        checkpoint = {
            'bert_state_dict': Embedding_model.state_dict(),
            'PLE_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
        }
        early_stopping(total_accuracy, epoch, checkpoint)

        if early_stopping.early_stop:
            print("早停！训练停止。")
            break
        scheduler.step()
        torch.cuda.empty_cache()

        if early_stopping.update == True:
            p_max_acc = per_result['accuracy']

            # ie = per_result['I/E_acc']
            # ns = per_result['N/S_acc']
            # tf = per_result['T/F_acc']
            # pj = per_result['P/J_acc']

            EXT = per_result['ext_acc']
            NEU = per_result['neu_acc']
            AGR = per_result['agr_acc']
            CON = per_result['con_acc']
            OPN = per_result['opn_acc']

            p_f1 = per_result['f1']
            e_max_acc = emo_result['accuracy']
            e_f1 = emo_result['f1']
        if early_stopping.early_stop:
            print("早停！训练停止。")
            break

    K_per_metrics['accuracy'].append(p_max_acc)

    # K_per_metrics['IE_acc'].append(ie)
    # K_per_metrics['NS_acc'].append(ns)
    # K_per_metrics['TF_acc'].append(tf)
    # K_per_metrics['PJ_acc'].append(pj)

    K_per_metrics['EXT_acc'].append(EXT)
    K_per_metrics['NEU_acc'].append(NEU)
    K_per_metrics['AGR_acc'].append(AGR)
    K_per_metrics['CON_acc'].append(CON)
    K_per_metrics['OPN_acc'].append(OPN)

    K_per_metrics['f1'].append(p_f1)

    K_emo_metrics['accuracy'].append(e_max_acc)
    K_emo_metrics['f1'].append(e_f1)

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