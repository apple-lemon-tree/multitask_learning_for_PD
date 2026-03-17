import argparse

import numpy as np
from sklearn.model_selection import StratifiedKFold

from tqdm import tqdm

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from transformers import AutoTokenizer

from data_loader.data_sentic_load1 import *
from MultitaskModels.PLE2 import *
from myutils.EarlyStop import EarlyStopping
from myutils.label_process import multilabel_process
from myutils.AffectiveSpaceFeatureExtractor import AffectiveFeatureExtractor
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

set_all_seeds(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

pnd_class_freq = [
    2947, 2602,3688, 3190, 5159
]

parser = argparse.ArgumentParser()

parser.add_argument("--checkpoint_path",type=str,default="/hy-tmp/check_points/essays_isear_checkpoint.pth",help="检查点保存路径")
parser.add_argument("--model_url",type=str,default="/hy-tmp/models/bert-base-uncased",help="分词和训练的模型")
parser.add_argument("--batch_size",type=int,default=16,help="批量大小")
parser.add_argument("--k_fold",type=int,default=5,help="交叉验证折数")
parser.add_argument("--personality_DataName",type=str,default="essays",help="人格任务的数据集名称")
parser.add_argument("--emotion_DataName",type=str,default="isear",help="情感任务的数据集名称")
parser.add_argument("--random_seed",type=int,default=42,help="随机种子")
parser.add_argument("--epoch",type=int,default=50,help="训练次数")
parser.add_argument("--learning_rate1",type=float,default=5e-6,help="学习率1")
parser.add_argument("--learning_rate2",type=float,default=1e-4,help="学习率2")
parser.add_argument("--eps",type=float,default=1e-8,help="数值稳定参数")
parser.add_argument("--patience",type=int,default=5,help="早停忍耐度")
parser.add_argument("--personality_url",type=str,default="/hy-tmp/dataset/Essays/回译/en2fr2de2en_bt_essays.csv",help="人格数据集路径")
parser.add_argument("--emotion_url",type=str,default="/hy-tmp/dataset/ISEAR/原始数据/ISEAR_1.csv",help="情感数据集路径")
parser.add_argument("--Senticnet8_url",type=str,default="/hy-tmp/dataset/affectivespace/affective.csv",help="情感数据集路径")
parser.add_argument("--pnd_max_length",type=int,default=512,help="人格数据截断长度")
parser.add_argument("--emo_max_length",type=int,default=128,help="情感数据截断长度")
parser.add_argument("--use_senticnet_feature",type=bool,default=True,help="是否使用SenticNet提取额外的100维特征")
parser.add_argument("--step_size",type=int,default=10,help="学习率衰减频率")
parser.add_argument("--gamma",type=float,default=0.1,help="学习率衰减倍数")

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

K_per_metrics= {'EXT_acc':[],'NEU_acc':[],'AGR_acc':[],'CON_acc':[],'OPN_acc':[],'accuracy': [],'f1': []}
K_emo_metrics = {'accuracy': [],'f1': []}

# 情感特征提取器
extractor = AffectiveFeatureExtractor(args.Senticnet8_url) if args.use_senticnet_feature else None

# 加载数据
p_texts, p_labels = read_data(args.personality_url,args.personality_DataName)
e_texts, e_labels = read_data(args.emotion_url,args.emotion_DataName)

# 加载dataloader
pnd_train_dataloader,pnd_val_dataloader,pnd_test_dataloader = create_dataloaders(
    batch_size=args.batch_size,
    texts=p_texts,
    labels=p_labels,
    tokenizer=tokenizers,
    max_length=args.pnd_max_length,
    stride=384,
    extractor=extractor,
    seed=args.random_seed,
)
emo_train_dataloader,emo_val_dataloader,emo_test_dataloader = create_dataloaders(
    batch_size=args.batch_size,
    texts=e_texts,
    labels=e_labels,
    tokenizer=tokenizers,
    max_length=args.emo_max_length,
    stride=384,
    extractor=extractor,
    seed=args.random_seed,
)
# 加载嵌入模型
Embedding_model = TextEmbedding(args.model_url).to(device)
# 加载主干模型
model = PLE(
    input_dim=768,
    expert_dim=768,
).to(device)

# 计算参数
parameters = list(model.parameters()) + list(Embedding_model.parameters())
print("训练参数量：", sum(p.numel() for p in parameters if p.requires_grad))

# 定义分层学习率
optimizer = optim.AdamW([
    {'params': Embedding_model.parameters(), 'lr': args.learning_rate1},
    {'params': model.shared_params(), 'lr': args.learning_rate2},
    {'params': model.tower1.parameters(), 'lr': args.learning_rate2},
    {'params': model.tower2.parameters(), 'lr': args.learning_rate2},
])
# 学习率衰减
scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

# 早停策略
early_stopping = EarlyStopping(patience=args.patience, verbose=True, path=args.checkpoint_path)


pnd_iter = iter(pnd_train_dataloader)
emo_iter = iter(emo_train_dataloader)

pnd_val_iter = iter(pnd_val_dataloader)
emo_val_iter = iter(emo_val_dataloader)

pnd_test_iter = iter(pnd_test_dataloader)
emo_test_iter = iter(emo_test_dataloader)

train_MAX,train_MIN = max(len(pnd_train_dataloader),len(emo_train_dataloader)),min(len(pnd_train_dataloader),len(emo_train_dataloader))
val_MAX,val_MIN = max(len(pnd_val_dataloader),len(emo_val_dataloader)),min(len(pnd_val_dataloader),len(emo_val_dataloader))
test_MAX,test_MIN= max(len(pnd_test_dataloader), len(emo_test_dataloader)), min(len(pnd_test_dataloader), len(emo_test_dataloader))


p_max_acc, e_max_acc, p_f1, e_f1 = 0.0, 0.0, 0.0, 0.0
EXT, NEU, AGR, CON, OPN = 0.0, 0.0, 0.0, 0.0, 0.0
ie, ns, tf, pj = 0.0, 0.0, 0.0, 0.0
for epoch in range(args.epoch):
    print(f"Epoch:{epoch + 1}")
    # 训练阶段
    p_train_loss, e_train_loss = 0.0, 0.0
    Embedding_model.train()
    model.train()
    for i in tqdm(range(train_MAX),desc="Train..."):
        optimizer.zero_grad()
        try:
            pnd_batch = next(pnd_iter)
        except StopIteration:
            pnd_iter = iter(pnd_train_dataloader)
            pnd_batch = next(pnd_iter)

        try:
            emo_batch = next(emo_iter)
        except StopIteration:
            emo_iter = iter(emo_train_dataloader)
            emo_batch = next(emo_iter)

        input_id1 = pnd_batch['input_ids'].to(device)
        attention_mask1 = pnd_batch['attention_mask'].to(device)
        mapping1 = pnd_batch['mapping']
        labels1 = pnd_batch['labels'].to(device)
        sentic_fea1 = pnd_batch['sentic_fea'].to(device)

        input_id2, attention_mask2, mapping2, labels2, sentic_fea2 = (
            value.to(device) for value in emo_batch.values()
        )

        pnd_embeddings = Embedding_model(input_ids1,attention_mask1,mapping1,sentic_fea1)
        emo_embeddings = Embedding_model(input_ids2,attention_mask2,mapping2,sentic_fea2)

        pnd_output, emo_output = model(pnd_embeddings,emo_embeddings)

        loss_1 = pnd_criterion(pnd_output, labels1)
        loss_2 = emo_criterion(emo_output, labels2)
        losses = torch.stack((loss_1,loss_2))
        weighted_loss, extra_outputs = weighting_method.backward(
            losses=losses,
            shared_parameters=list(Embedding_model.parameters())+model.shared_params(),
            task_specific_parameters=...
        )
        # loss = loss_1 + loss_2

        p_train_loss += loss_1.item()
        e_train_loss += loss_2.item()

        # loss.backward()
        nn.utils.clip_grad_norm_(Embedding_model.parameters(), max_norm=1.0, norm_type=2)
        nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.0,norm_type=2)
        optimizer.step()
    print("权重：",extra_outputs['weights'])
    print(f"损失情况： 人格任务损失:{p_train_loss},情感任务损失:{e_train_loss}")


    # 验证阶段
    p_val_loss, e_val_loss = 0.0, 0.0
    Embedding_model.eval()
    model.eval()
    with torch.no_grad():
        pnd_pred, pnd_label, emo_pred, emo_label = [], [], [], []

        for i in tqdm(range(val_MIN), desc="PND_Eval"):
            try:
                pnd_val_batch = next(pnd_val_iter)
            except StopIteration:
                pnd_val_iter = iter(pnd_val_dataloader)
                pnd_val_batch = next(pnd_val_iter)

            try:
                emo_val_batch = next(emo_val_iter)
            except StopIteration:
                emo_val_iter = iter(emo_val_dataloader)
                emo_val_batch = next(emo_val_iter)

            input_ids1 = pnd_val_batch['input_ids'].to(device)
            attention_mask1 = pnd_val_batch['attention_mask'].to(device)
            labels1 = pnd_val_batch['labels'].to(device)
            map1 = pnd_val_batch['mapping']
            sentic_fea1 = pnd_val_batch['sentic_fea'].to(device)

            input_ids2 = emo_val_batch['input_ids'].to(device)
            attention_mask2 = emo_val_batch['attention_mask'].to(device)
            labels2 = emo_val_batch['labels'].to(device)
            map2 = emo_val_batch['mapping']
            sentic_fea2 = emo_val_batch['sentic_fea'].to(device)

            pnd_embeddings = Embedding_model(input_ids1, attention_mask1,mapping1, sentic_fea1)
            emo_embeddings = Embedding_model(input_ids2, attention_mask2,mapping2, sentic_fea2)
            out1, out2 = model(pnd_embeddings, emo_embeddings)
            p_val_loss += pnd_criterion(out1, labels1).item()
            sigmoid = nn.Sigmoid()
            probs = sigmoid(out1)
            predictions = (probs > 0.5).to(torch.int8)
            pnd_pred.extend(predictions.cpu().numpy())
            pnd_label.extend(labels1.cpu().numpy())

        for i in tqdm(range(val_MAX), desc="EMO_Eval"):
            try:
                pnd_val_batch = next(pnd_val_iter)
            except StopIteration:
                pnd_val_iter = iter(pnd_val_dataloader)
                pnd_val_batch = next(pnd_val_iter)

            try:
                emo_val_batch = next(emo_val_iter)
            except StopIteration:
                emo_val_iter = iter(emo_val_dataloader)
                emo_val_batch = next(emo_val_iter)

            input_ids1 = pnd_val_batch['input_ids'].to(device)
            attention_mask1 = pnd_val_batch['attention_mask'].to(device)
            labels1 = pnd_val_batch['labels'].to(device)
            map1 = pnd_val_batch['mapping']
            sentic_fea1 = pnd_val_batch['sentic_fea'].to(device)

            input_ids2 = emo_val_batch['input_ids'].to(device)
            attention_mask2 = emo_val_batch['attention_mask'].to(device)
            labels2 = emo_val_batch['labels'].to(device)
            map2 = emo_val_batch['mapping']
            sentic_fea2 = emo_val_batch['sentic_fea'].to(device)

            pnd_embeddings = Embedding_model(input_ids1, attention_mask1, sentic_fea1)
            emo_embeddings = Embedding_model(input_ids2, attention_mask2, sentic_fea2)
            out1, out2 = model(pnd_embeddings, emo_embeddings)
            e_val_loss += emo_criterion(out1, labels1).item()
            predictions = torch.argmax(out2, dim=1)

            emo_pred.extend(predictions.cpu().numpy())
            emo_label.extend(labels2.cpu().numpy())

    ext, neu, agr, con, opn = calculate_per_dim_accuracy(pnd_label, pnd_pred)
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
    print(f"损失情况： 人格任务损失={p_val_loss},情感任务损失={e_val_loss}")
    print(per_result)
    print(emo_result)
    total_accuracy = 0.7 * per_result['accuracy'] + 0.3 * emo_result['accuracy']

    checkpoint = {
        'bert_state_dict': Embedding_model.state_dict(),
        'PLE_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
    }
    early_stopping(total_accuracy,epoch,checkpoint)

    if early_stopping.early_stop:
        print("早停！训练停止。")
        break
    scheduler.step()
    torch.cuda.empty_cache()


# 测试阶段
# 加载最佳模型
best_checkpoint = torch.load(args.best_checkpoint)
Embedding_model.load_state_dict(best_checkpoint['bert_state_dict'])
model.load_state_dict(best_checkpoint['PLE_state_dict'])
Embedding_model.eval()
model.eval()
with torch.no_grad():
    pnd_pred, pnd_label, emo_pred, emo_label = [], [], [], []

    for i in tqdm(range(test_MIN), desc="PND_Test"):
        try:
            pnd_test_batch = next(pnd_test_iter)
        except StopIteration:
            pnd_test_iter = iter(pnd_test_dataloader)
            pnd_test_batch = next(pnd_test_iter)

        try:
            emo_test_batch = next(emo_test_iter)
        except StopIteration:
            emo_test_iter = iter(emo_test_dataloader)
            emo_test_batch = next(emo_test_iter)

        input_ids1 = pnd_test_batch['input_ids'].to(device)
        attention_mask1 = pnd_test_batch['attention_mask'].to(device)
        labels1 = pnd_test_batch['labels'].to(device)
        map1 = pnd_test_batch['mapping']
        sentic_fea1 = pnd_test_batch['sentic_fea'].to(device)

        input_ids2 = emo_test_batch['input_ids'].to(device)
        attention_mask2 = emo_test_batch['attention_mask'].to(device)
        labels2 = emo_test_batch['labels'].to(device)
        map2 = emo_test_batch['mapping']
        sentic_fea2 = emo_test_batch['sentic_fea'].to(device)

        pnd_embeddings = Embedding_model(input_ids1, attention_mask1,mapping1, sentic_fea1)
        emo_embeddings = Embedding_model(input_ids2, attention_mask2,mapping2, sentic_fea2)
        out1, out2 = model(pnd_embeddings, emo_embeddings)
        sigmoid = nn.Sigmoid()
        probs = sigmoid(out1)
        predictions = (probs > 0.5).to(torch.int8)
        pnd_pred.extend(predictions.cpu().numpy())
        pnd_label.extend(labels1.cpu().numpy())

    for i in tqdm(range(test_MAX), desc="EMO_Test"):
        try:
            pnd_test_batch = next(pnd_test_iter)
        except StopIteration:
            pnd_test_iter = iter(pnd_test_dataloader)
            pnd_test_batch = next(pnd_test_iter)

        try:
            emo_test_batch = next(emo_test_iter)
        except StopIteration:
            emo_test_iter = iter(emo_test_dataloader)
            emo_test_batch = next(emo_test_iter)

        input_ids1 = pnd_test_batch['input_ids'].to(device)
        attention_mask1 = pnd_test_batch['attention_mask'].to(device)
        labels1 = pnd_test_batch['labels'].to(device)
        map1 = pnd_test_batch['mapping']
        sentic_fea1 = pnd_test_batch['sentic_fea'].to(device)

        input_ids2 = emo_test_batch['input_ids'].to(device)
        attention_mask2 = emo_test_batch['attention_mask'].to(device)
        labels2 = emo_test_batch['labels'].to(device)
        map2 = emo_test_batch['mapping']
        sentic_fea2 = emo_test_batch['sentic_fea'].to(device)

        pnd_embeddings = Embedding_model(input_ids1, attention_mask1, sentic_fea1)
        emo_embeddings = Embedding_model(input_ids2, attention_mask2, sentic_fea2)
        out1, out2 = model(pnd_embeddings, emo_embeddings)
        predictions = torch.argmax(out2, dim=1)
        emo_pred.extend(predictions.cpu().numpy())
        emo_label.extend(labels2.cpu().numpy())

ext, neu, agr, con, opn = calculate_per_dim_accuracy(pnd_label, pnd_pred)
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
print("----- 模型测试结果 -----")
print(per_result)
print(emo_result)