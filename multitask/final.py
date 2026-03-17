import argparse
import random
from sklearn.model_selection import StratifiedKFold
from torch import nn
from tqdm import tqdm
from itertools import cycle

from sklearn.metrics import accuracy_score, f1_score
import torch.optim as optim
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from data_loader.load_per_isear_ori import *
from MultitaskModels.MyMultiTaskModel import MultitaskTextCNN,TextEmbedding
from myutils.EarlyStop import EarlyStopping
from myutils.label_process import multilabel_process
from myutils.calculate_per_dim_accuracy import calculate_per_dim_accuracy, calculate_mbti_accuracy
from myutils.weight_methods import NashMTL_CPU

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

parser.add_argument("--checkpoint_path",type=str,default="../checkpoint/kFoldAndTrans_essays_isear_checkpoint.pth",help="检查点保存路径")
parser.add_argument("--model_url",type=str,default="/hy-tmp/models/bert-base-uncased",help="分词和训练的模型")
parser.add_argument("--batch_size",type=int,default=16,help="批量大小")
parser.add_argument("--k_fold",type=int,default=5,help="交叉验证折数")
parser.add_argument("--personality_DataName",type=str,default="mbti",help="人格任务的数据集名称")
parser.add_argument("--emotion_DataName",type=str,default="isear",help="情感任务的数据集名称")
parser.add_argument("--random_seed",type=int,default=67373,help="随机种子")
parser.add_argument("--epoch",type=int,default=30,help="训练次数")
parser.add_argument("--learning_rate1",type=float,default=2e-6,help="bert学习率")
parser.add_argument("--learning_rate2",type=float,default=1e-4,help="textCNN及分类头学习率")
parser.add_argument("--weight_decay",type=float,default=0.01,help="权重衰减")
parser.add_argument("--eps",type=float,default=1e-8,help="数值稳定参数")
parser.add_argument("--patience",type=int,default=5,help="早停忍耐度")
parser.add_argument("--personality_url",type=str,default="/hy-tmp/dataset/MBTI/原始数据/多数类1.csv",help="人格数据集路径")
parser.add_argument("--emotion_url",type=str,default="/hy-tmp/dataset/ISEAR/原始数据/ISEAR_1.csv",help="情感数据集路径")
parser.add_argument("--pnd_max_length",type=int,default=512,help="人格数据截断长度")
parser.add_argument("--emo_max_length",type=int,default=64,help="情感数据截断长度")
parser.add_argument("--warmup_rate",type=int,default=0.1,help="warmup比例")
parser.add_argument("--step_size",type=int,default=5,help="阶梯衰减epoch数")
parser.add_argument("--gamma",type=float,default=0.5,help="")


args = parser.parse_args()

set_all_seeds(args.random_seed)

tokenizer = AutoTokenizer.from_pretrained(args.model_url)

k_fold = args.k_fold

p_text, p_labels = read_data(args.personality_url,args.personality_DataName)
e_text, e_labels = read_data(args.emotion_url,args.emotion_DataName)

skf = StratifiedKFold(n_splits=args.k_fold, shuffle=True,random_state=args.random_seed)

personality_criterion = nn.BCEWithLogitsLoss().to(device)
emotion_criterion = nn.CrossEntropyLoss().to(device)

weighting_method = NashMTL_CPU(n_tasks=2,device=device)

essays_K = skf.split(p_text, multilabel_process(p_labels))
isear_K = skf.split(e_text, e_labels)

K_per_metrics= {
    'EI_acc':[],'SN_acc':[],'TF_acc':[],'JP_acc':[],
    'accuracy': [],
    'EI_f1':[],'SN_f1':[],'TF_f1':[],'JP_f1':[],
    'macro-f1':[]
}
K_emo_metrics = {'accuracy': [],'f1': []}

for fold, ((p_train_index, p_test_index),(e_train_index, e_test_index)) in enumerate(zip(essays_K, isear_K)):
    print("*" * 50)
    print(f"第{fold + 1}折")
    print("*" * 50)

    pnd_train_dataloader, pnd_val_dataloader = create_dataloaders(
        batch_size=args.batch_size,
        text=p_text,
        labels=p_labels,
        tokenizer=tokenizer,
        max_length=args.pnd_max_length,
        train_idx=p_train_index,
        val_idx=p_test_index,
        seed=args.random_seed,
    )

    emo_train_dataloader, emo_val_dataloader = create_dataloaders(
        batch_size=args.batch_size,
        text=e_text,
        labels=e_labels,
        tokenizer=tokenizer,
        max_length=args.emo_max_length,
        train_idx=e_train_index,
        val_idx=e_test_index,
        seed=args.random_seed,
    )
    print("训练数据加载完毕...")
    EmbeddingModel = TextEmbedding(args.model_url).to(device)
    TextCNNModel = MultitaskTextCNN().to(device)
    parameters = list(TextCNNModel.parameters()) + list(EmbeddingModel.parameters())
    print("训练参数量：", sum(p.numel() for p in parameters if p.requires_grad))
    adapter_params = [p for p in list(EmbeddingModel.parameters()) if p.requires_grad]
    print("adapter训练参数量：", sum(p.numel() for p in adapter_params))

    optimizer_bert = optim.AdamW(EmbeddingModel.parameters(), lr=args.learning_rate1)
    optimizer_cnn = optim.AdamW(TextCNNModel.parameters(), lr=args.learning_rate2)

    num_training_steps = len(pnd_train_dataloader) * args.epoch
    num_warmup_steps = int(num_training_steps * args.warmup_rate)
    scheduler_bert = get_linear_schedule_with_warmup(
        optimizer_bert,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    scheduler_cnn = optim.lr_scheduler.StepLR(
        optimizer_cnn,
        step_size=args.step_size,
        gamma=args.gamma,
    )


    early_stopping = EarlyStopping(patience=args.patience, verbose=True)

    p_max_acc,e_max_acc, p_f1, e_f1 = 0.0, 0.0,0.0,0.0
    EI_acc, SN_acc, TF_acc, JP_acc = 0.0,0.0,0.0,0.0
    EI_f1, SN_f1, TF_f1, JP_f1 = 0.0, 0.0, 0.0, 0.0
    for epoch in range(args.epoch):
        print(f"Epoch:{epoch + 1}")

        EmbeddingModel.train()
        TextCNNModel.train()
        p_loss, i_loss = 0.0, 0.0
        cycled_emo_train_dataloader = cycle(emo_train_dataloader)
        for batch1 in tqdm(pnd_train_dataloader,desc="Train..."):

            batch2 = next(cycled_emo_train_dataloader)

            input_ids1 = batch1['input_ids'].to(device)
            attention_mask1 = batch1['attention_mask'].to(device)
            labels1 = batch1['labels'].to(device).float()
            sentic_fea1 = batch1['sentic_fea'].to(device)

            pnd_vec = EmbeddingModel(input_ids1,attention_mask1)

            input_ids2 = batch2['input_ids'].to(device)
            attention_mask2 = batch2['attention_mask'].to(device)
            labels2 = batch2['labels'].to(device)
            sentic_fea2 = batch2['sentic_fea'].to(device)

            emo_vec = EmbeddingModel(input_ids2,attention_mask2)

            pnd_output,emo_output = TextCNNModel(pnd_vec,emo_vec,sentic_fea1,sentic_fea2)

            loss1 = personality_criterion(pnd_output, labels1)
            loss2 = emotion_criterion(emo_output, labels2)
            losses = torch.stack(
                (
                    loss1,
                    loss2
                )
            )
            optimizer_bert.zero_grad()
            optimizer_cnn.zero_grad()

            weighted_loss, extra_outputs = weighting_method.backward(
                losses=losses,
                shared_parameters=list(EmbeddingModel.parameters()) + TextCNNModel.shared_parameters(),
                # task_specific_parameters=model.task_specific_parameters()
            )

            loss = loss1 + loss2

            p_loss += loss1.item()
            i_loss += loss2.item()

            # loss.backward()
            # nn.utils.clip_grad_norm_(EmbeddingModel.parameters(),max_norm=1.0,norm_type=2)
            # nn.utils.clip_grad_norm_(TextCNNModel.parameters(), max_norm=1.0, norm_type=2)
            optimizer_bert.step()
            optimizer_cnn.step()
            scheduler_bert.step()
        print(f"PND Train loss:{p_loss},EMO Train loss:{i_loss}")

        EmbeddingModel.eval()
        TextCNNModel.eval()
        cycled_emo_val_dataloader = cycle(emo_val_dataloader)
        with (torch.no_grad()):

            pnd_pred = []
            pnd_label = []
            emo_pred = []
            emo_label = []
            for batch1 in tqdm(pnd_val_dataloader,desc="Validation..."):

                batch2 = next(cycled_emo_val_dataloader)

                input_ids1 = batch1['input_ids'].to(device)
                attention_mask1 = batch1['attention_mask'].to(device)
                labels1 = batch1['labels'].to(device)
                sentic_fea1 = batch1['sentic_fea'].to(device)
                pnd_vec = EmbeddingModel(input_ids1,attention_mask1)

                input_ids2 = batch2['input_ids'].to(device)
                attention_mask2 = batch2['attention_mask'].to(device)
                labels2 = batch2['labels'].to(device)
                sentic_fea2 = batch2['sentic_fea'].to(device)
                emo_vec = EmbeddingModel(input_ids2,attention_mask2)

                pnd_out,emo_out = TextCNNModel(pnd_vec,emo_vec,sentic_fea1,sentic_fea2)
                pnd_out = torch.sigmoid(pnd_out)
                predictions1 = (pnd_out > 0.5).to(torch.int8)
                pnd_pred.extend(predictions1.cpu().numpy())
                pnd_label.extend(labels1.cpu().numpy())

                predictions2 = torch.argmax(emo_out, dim=1)
                emo_pred.extend(predictions2.cpu().numpy())
                emo_label.extend(labels2.cpu().numpy())

            EI, SN, TF, JP = calculate_mbti_accuracy(pnd_label, pnd_pred)
            f1_scores_per_dim = f1_score(pnd_label, pnd_pred, average=None, zero_division=0)
            per_result = {
                'EI_acc': EI,
                'SN_acc': SN,
                'TF_acc': TF,
                'JP_acc': JP,
                'accuracy': accuracy_score(np.concatenate(pnd_label), np.concatenate(pnd_pred)),
                'EI_f1': f1_scores_per_dim[0],
                'SN_f1': f1_scores_per_dim[1],
                'TF_f1': f1_scores_per_dim[2],
                'JP_f1': f1_scores_per_dim[3],
                'macro-f1': f1_score(pnd_label, pnd_pred, average='macro')
            }
            emo_result = {
                'accuracy': accuracy_score(emo_label, emo_pred),
                'f1': f1_score(emo_label, emo_pred, average='macro'),
            }
            formatted_result1 = {k: f"{v:.4f}" for k, v in per_result.items()}
            print(formatted_result1)
            formatted_result2 = {k: f"{v:.4f}" for k, v in emo_result.items()}
            print(formatted_result2)
            total_accuracy =  0.7 * per_result['accuracy'] + 0.3 * emo_result['accuracy']
            early_stopping(total_accuracy, epoch, checkpoint=None)

        if early_stopping.update == True:

            EI_acc = per_result['EI_acc']
            SN_acc = per_result['SN_acc']
            TF_acc = per_result['TF_acc']
            JP_acc = per_result['JP_acc']
            p_max_acc = per_result['accuracy']

            EI_f1 = per_result['EI_f1']
            SN_f1 = per_result['SN_f1']
            TF_f1 = per_result['TF_f1']
            JP_f1 = per_result['JP_f1']
            p_f1 = per_result['macro-f1']

            e_max_acc = emo_result['accuracy']
            e_f1 = emo_result['f1']
        if early_stopping.early_stop:
            print("早停！训练停止。")
            break

        scheduler_cnn.step()


    K_per_metrics['EI_acc'].append(EI_acc)
    K_per_metrics['SN_acc'].append(SN_acc)
    K_per_metrics['TF_acc'].append(TF_acc)
    K_per_metrics['JP_acc'].append(JP_acc)

    K_per_metrics['accuracy'].append(p_max_acc)

    K_per_metrics['EI_f1'].append(EI_f1)
    K_per_metrics['SN_f1'].append(SN_f1)
    K_per_metrics['TF_f1'].append(TF_f1)
    K_per_metrics['JP_f1'].append(JP_f1)

    K_per_metrics['macro-f1'].append(p_f1)

    K_emo_metrics['accuracy'].append(e_max_acc)
    K_emo_metrics['f1'].append(e_f1)

print("personality metrics:")
for key, value in K_per_metrics.items():
    print(f"{key}:{value}")
    print(f"{key}: {np.mean(value):.4f}")
print("emotion metrics:")
for key, value in K_emo_metrics.items():
    print(f"{key}:{value}")
    print(f"{key}: {np.mean(value):.4f}")
