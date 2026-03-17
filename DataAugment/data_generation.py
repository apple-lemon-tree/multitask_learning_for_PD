import argparse
import random
import re
from collections import Counter

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import jaccard_score
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, GPT2Config, get_linear_schedule_with_warmup, AutoTokenizer

from data_utils import PersonalityDataset
from mymodel import MyModel, Filter
BERT_URL = "/hy-tmp/models/bert-base-uncased"
GPT2_URL = "/hy-tmp/models/gpt2"


def top_k_top_p_filtering(logits, top_k=0, top_p=1.0, filter_value=-float("inf")):
    """Apply top-k and/or nucleus (top-p) filtering to logits."""
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        values, _ = torch.topk(logits, top_k)
        min_values = values[..., -1, None]
        logits = torch.where(logits < min_values, torch.full_like(logits, filter_value), logits)

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
        indices_to_remove.scatter_(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
        logits = logits.masked_fill(indices_to_remove, filter_value)

    return logits


def generate_with_prefix(gpt_model, prefix, prompt_ids, max_new_tokens, top_k, top_p, device):
    """Sampling loop using `past_key_values` prefix and full prompt."""
    # 1) 先用整句 prompt 融合 prefix，得到新的 past
    prompt_ids = prompt_ids.to(device)
    outputs = gpt_model(input_ids=prompt_ids, past_key_values=prefix)
    past = outputs.past_key_values
    generated = prompt_ids

    # 2) 之后每一步只喂新采样出的一个 token
    cur_input_ids = prompt_ids[:, -1:]
    for _ in range(max_new_tokens):
        outputs = gpt_model(input_ids=cur_input_ids, past_key_values=past)
        logits = outputs.logits[:, -1, :]
        filtered_logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
        probs = torch.softmax(filtered_logits, dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=1)

        generated = torch.cat([generated, next_tokens], dim=-1)
        cur_input_ids = next_tokens
        past = outputs.past_key_values

    return generated

def init_hyparameters():

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="Kaggle")
    parser.add_argument('--num_train_epochs', type=int, default=30)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument('--train', type=bool, default=True)
    parser.add_argument('--eval', type=bool, default=True)
    parser.add_argument('--bert_lr', type=float, default=1e-5)
    parser.add_argument('--bert_max_len', type=int, default=512)
    parser.add_argument('--gpt_max_len', type=int, default=512)
    parser.add_argument('--gpt_top_p', type=float, default=0.95)
    parser.add_argument('--gpt_top_k', type=int, default=50)

    args = parser.parse_args()
    args.gpt_ckpt_dir = f"output/{args.dataset}/gpt_ckpt.pt"
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.sep = ", "
    args.class_num = 5 if args.dataset == "Essays" else 8

    return args

class Model(nn.Module):
    def __init__(self,dataset,gpt_tokenizer,train_dataset,alpha,beta,do_train=False):
        super(Model, self).__init__()
        self.dataset = dataset
        self.tokenizer = gpt_tokenizer
        self.alpha = alpha
        self.beta = beta
        self.do_train = do_train
        self.model = MyModel(
            config=GPT2Config.from_pretrained(GPT2_URL),
            preseqlen=10,
            mid_dim=512,
            dataset=self.dataset,
            tokenizer=self.tokenizer,
            use_adapter=False,
            content_embeddings=train_dataset.content_embeddings)

        print(f"训练参数量：{sum(p.numel() for p in self.model.parameters() if p.requires_grad)}")

    def forward(self, batch):
        labels = batch['target_ids']
        labels[labels[:,:] == self.tokenizer.pad_token_id] = -100

        attention_mask = batch['source_mask']
        pad_seq_mask = torch.sum(attention_mask[:,1:], dim=1).gt(0)

        input_ids = batch['source_ids'][pad_seq_mask]
        labels = labels[pad_seq_mask]
        label_list = batch['labels'][pad_seq_mask]

        bert_ids = batch['bert_ids'][pad_seq_mask]
        bert_mask = batch['bert_mask'][pad_seq_mask]
        clusters = batch['clusters'][pad_seq_mask]

        outputs, loss2 = self.model(input_ids=input_ids,
                                    labels=labels,
                                    label_list=label_list,
                                    bert_ids=bert_ids,
                                    bert_mask=bert_mask,
                                    clusters=clusters,
                                    do_train=self.do_train)

        return outputs,loss2

    def compute_loss(self, batch, alpha=0.5, beta=0.5):
        outputs, loss2 = self.forward(batch)
        loss = outputs.loss
        label_loss = loss2["label"]
        content_loss = loss2["content"]

        total_loss = loss + alpha * label_loss + beta * content_loss
        return total_loss, label_loss, content_loss

def kaggle_data_processing(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = text.encode('ascii', 'ignore').decode('ascii')
    if text.startswith("'"):
        text = text[1:-1]

    MBTIs = ('INTJ', 'INTP', 'INFP', 'ENTP', 'ISTP', 'ISFP', 'ESTJ', 'ISTJ',
             'ESTP', 'ISFJ', 'ENFP', 'ESFP', 'ESFJ', 'ENFJ', 'INFJ', 'ENTJ')
    token = '<mask>'
    for mbti in MBTIs:
        if mbti in text:
            text = text.replace(mbti.lower(), token)

    return text

def train_epoch(model, dataloader, optimizer, scheduler, epoch, device):
    model.train()
    total_loss = 0
    total_label_loss = 0
    total_content_loss = 0
    num_batches = 0

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for step,batch in enumerate(progress_bar):
        batch = {k:v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        loss, l_loss, c_loss = model.compute_loss(batch)

        if loss is None or (isinstance(loss,torch.Tensor) and loss.item() == 0):
            continue

        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        num_batches += 1
        total_loss += loss.item() if isinstance(loss, torch.Tensor) else loss
        total_label_loss += l_loss.item() if isinstance(l_loss, torch.Tensor) else l_loss
        total_content_loss += c_loss.item() if isinstance(c_loss, torch.Tensor) else c_loss

    if num_batches == 0: return 0,0,0

    avg_loss = total_loss / num_batches
    avg_label_KL = total_label_loss / num_batches
    avg_content_KL = total_content_loss / num_batches
    print(f"\nEpoch {epoch} Train Loss: {avg_loss:.4f}, Label: {avg_label_KL:.4f}, Content: {avg_content_KL:.4f}")
    return avg_loss, avg_label_KL, avg_content_KL

def save_checkpoint(save_dir, epoch, model,optimizer):
    ckpt = {
        'model_state_dict': model.model.state_dict(),
        'alpha': model.alpha,
        'beta': model.beta,
        'optimizer': optimizer.state_dict(),
        'dataset_name': model.dataset,
        'epoch': epoch,
    }

    torch.save(ckpt, save_dir)
    print(f"检查点保存在 {save_dir}")


def validate_epoch(model,dataloader,epoch,device,alpha,beta):
    model.eval()
    total_loss = 0
    total_label_loss = 0
    total_content_loss = 0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Validation Epoch {epoch}"):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            loss, l_loss, c_loss = model.compute_loss(batch, alpha=alpha, beta=beta)
            total_label_loss += l_loss.item() if isinstance(l_loss, torch.Tensor) else l_loss
            total_content_loss += c_loss.item() if isinstance(c_loss, torch.Tensor) else c_loss

            total_loss += loss.item()
            num_batches += 1

    if num_batches == 0: return 0, 0, 0

    avg_loss = total_loss / num_batches
    avg_l = total_label_loss / num_batches
    avg_c = total_content_loss / num_batches
    print(f"\nEpoch {epoch} Val Loss: {avg_loss:.4f}, Label: {avg_l:.4f}, Content: {avg_c:.4f}")
    return avg_loss, avg_l, avg_c


if __name__ == '__main__':

    args = init_hyparameters()
    device = args.device
    gpt_tokenizer = AutoTokenizer.from_pretrained(GPT2_URL)
    special_tokens_dict = {"pad_token": "[PAD]"}
    gpt_tokenizer.add_special_tokens(special_tokens_dict)
    gpt_tokenizer.padding_side = "right"


    train_dataset = PersonalityDataset(tokenizer=gpt_tokenizer,data_type='train',data_dir=args.dataset,
                                       max_len=512 if args.dataset=="Essays" else 64,batch_size=args.batch_size,
                                       device=args.device,class_num=args.class_num)
    val_dataset = train_dataset

    train_loader = DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True)
    val_loader = DataLoader(val_dataset,batch_size=args.batch_size,shuffle=False)

    if args.train:
        print("=====开始训练=====")
        model = Model(args.dataset, gpt_tokenizer, train_dataset, args.alpha, args.beta, do_train=True)
        model.to(device)

        optimizer = AdamW(model.parameters(), lr=args.bert_lr, weight_decay=0.0)

        # ==== 尝试从已有检查点恢复 ====
        start_epoch = 0
        best_val_loss = float("inf")
        patience_counter = 0
        max_patience = 3
        try:
            ckpt = torch.load(args.gpt_ckpt_dir, map_location=device)
            print(f"检测到已有检查点，尝试从 {ckpt['epoch'] + 1} 轮继续训练")
            model.model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer"])
            # 如果你希望 alpha/beta 也从 ckpt 中继续，用下面两行替换当前 args 的值
            # args.alpha, args.beta = ckpt.get("alpha", args.alpha), ckpt.get("beta", args.beta)
            start_epoch = ckpt.get("epoch", -1) + 1
        except (FileNotFoundError, RuntimeError, KeyError):
            print("未找到或无法加载检查点，将从头开始训练。")

        num_training_steps = len(train_loader) * (args.num_train_epochs - start_epoch)
        num_warmup_steps = int(0.05 * num_training_steps) if num_training_steps > 0 else 0

        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=max(num_training_steps, 1)
        )

        for epoch in range(start_epoch, args.num_train_epochs):
            t_loss,t_l,t_c = train_epoch(model,train_loader,optimizer,scheduler,epoch, device)
            v_loss, v_l, v_c = validate_epoch(model, val_loader, epoch, device,args.alpha,args.beta)
            if v_loss < best_val_loss:
                best_val_loss = v_loss
                save_checkpoint(args.gpt_ckpt_dir,epoch,model,optimizer)
                patience_counter = 0
            else:
                patience_counter += 1
                print(f"Patience: {patience_counter}/{max_patience}")

            if patience_counter >= max_patience:
                print("早停")
                break

    if args.eval:
        val_dataset = train_dataset
        if args.dataset == "Essays":
            labels = ['extraversion','neuroticism','agreeableness','conscientiousness','openness']
            label2index = {l: i for i, l in enumerate(labels)}
            # {'ext':0,'neu':1,'agr':2,'con':3,'opn':4}
        elif args.dataset == "Kaggle":
            labels = ['introversion','extraversion','intuition','sensing','feeling','thinking','perception','judgment']
            label2index = {l: i for i, l in enumerate(labels)}
            # {'i':0,'e':1,'n':2,...}

        df = pd.read_csv(f"data/{args.dataset}/prompts.txt",sep="\t",header=None)
        labels = df.iloc[:,0].values.tolist()
        prompts = [" ".join(lab.split()[7:]) for lab in labels]
        print(prompts)
        prompt_counter = dict(Counter(prompts))
        prompt_set = sorted(list(set(prompts)))
        label_lists = []

        for prompt in prompts:
            ori_list = [0] * args.class_num
            for label in prompt.split(args.sep):
                ori_list[label2index[label]] = 1
            label_lists.append(ori_list)

        bert_tokenizer = AutoTokenizer.from_pretrained(BERT_URL)
        filter = Filter(args.class_num)
        filter.load_state_dict(torch.load(f"filter_model/{args.dataset}/best_filter.pt"))
        filter.to(device)

        checkpoint = torch.load(args.gpt_ckpt_dir, map_location=device)
        alpha, beta, dataset_name = checkpoint['alpha'], checkpoint['beta'], checkpoint['dataset_name']
        model = Model(
            dataset=dataset_name,
            gpt_tokenizer=gpt_tokenizer,
            train_dataset=val_dataset,
            alpha=args.alpha,
            beta=args.beta,
        )
        model.model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)

        model.eval()

        final = []
        final_dict = {}

        with torch.no_grad():
            for p, total_num in prompt_counter.items():
                print(f"生成文本中")
                prompt_list = []
                label_list = [0] * args.class_num
                for label in p.split(args.sep):
                    label_list[label2index[label]] = 1
                for idx in tqdm(range(3 * total_num // 10 + 1)):
                    if args.dataset == "Essays":
                        input_seq = "A personal essay that shows author's personality traits of " + p + " :"
                    elif args.dataset == "Kaggle":
                        input_seq = "Some personal sentences that shows the personality traits of " + p + " :"
                    # print(f"input_seq: {input_seq}")
                    tokenized = gpt_tokenizer(input_seq,return_tensors="pt").to(device)
                    input_ids = tokenized["input_ids"].repeat(10,1).to(device)
                    labels = torch.tensor(label_list).unsqueeze(0).repeat(10,1).to(device)
                    prefix, _ = model.model.get_prompt(batch_size=10,labels=labels,do_train=False,)
                    # print(f"input_ids.shape: {input_ids.shape}")  (10, input_seq分词后的token数)，
                    if isinstance(prefix,tuple):
                        # (10,12 heads,seqlen,64) 也就是(330行设定的repeat数,gpt2注意头数,（单个）label的prefix前缀长度,gpt的每个注意头的嵌入维度 768/12=64)
                        stack_tensor = torch.stack(prefix)
                        #print(f"prefix.shape: {stack_tensor.shape}")# torch.Size([12, 2, 10, 12, 10, 64])

                    else:
                        print(f"prefix.shape:{prefix.shape}")

                    # print(f"len(prefix): {len(prefix)}") 12

                    # 使用自定义采样循环，显式利用 prefix 和整句 prompt
                    beam_outputs = generate_with_prefix(
                        gpt_model=model.model.gpt,
                        prefix=prefix,
                        prompt_ids=input_ids,
                        max_new_tokens=args.gpt_max_len,
                        top_k=args.gpt_top_k,
                        top_p=args.gpt_top_p,
                        device=device,
                    )
                    prompt_list.append(
                        [p + '\t' + model.tokenizer.decode(beam_output, skip_special_tokens=True).replace("\n", "").strip()
                         for beam_output in beam_outputs])

                final_dict[p] = sum(prompt_list, [])

            sorted_dicts = {}

            for p,sen_list in final_dict.items():
                temp = {}
                label_list = [0] * args.class_num
                for label in p.split(args.sep):
                    #print(f"label: {label}")
                    label_list[label2index[label]] = 1
                for sen in sen_list:
                    tokenized = bert_tokenizer(" ".join(sen.split(" :")[1:]).strip(),
                                               return_tensors="pt",max_length=args.bert_max_len,
                                               padding="max_length",truncation=True)
                    out = filter(tokenized["input_ids"].to(device), tokenized["attention_mask"].to(device))
                    processed_out = torch.sigmoid(out)
                    prediction = (processed_out > 0.5).int().squeeze(0).cpu().tolist()

                    temp[sen] = jaccard_score(label_list,prediction,zero_division=0)
                temp = sorted(temp.items(), key=lambda x: x[1], reverse=True)
                p_list = [(p, x[0]) for x in temp]
                sorted_dicts[p] = p_list

            for l, num in prompt_counter.items():
                final.append(sorted_dicts[l][: num])

            final = sum(final, [])
            random.shuffle(final)
            with open(f"augmentation/aug_{args.dataset}.txt", "w") as w:
                for p, generated in final:
                    w.write(generated + "\n")

