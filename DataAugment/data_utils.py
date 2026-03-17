import os
import re
from collections import Counter

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

SEED = 42
BERT_URL = "/hy-tmp/models/bert-base-uncased"
GPT2_URL = "/hy-tmp/models/gpt2"
def get_data(path, data_type):
    sents, labels, label_idx = [], [], []
    df = pd.read_csv(path, sep="\t", header=None)

    sents_array = df.iloc[:, 1].values.tolist()
    label_array = df.iloc[:, 0].values.tolist()
    label_array = [eval(a) for a in label_array]

    for temp, sen in zip(label_array, sents_array):
        if "train" in data_type:
            if sum(temp) > 0:# 排除掉标签为[0, 0, 0, 0, 0]的数据
                labels.append(temp)
            else:
                continue
        else:
            labels.append(temp)
        sen = sen.split()
        if sen:
            sents.append(" ".join(sen))
    # print(f"len(sents_array): {len(sents_array)}") 1973
    # print(f"len(label_array): {len(label_array)}") 1973
    # print(f"len(sents): {len(sents)})") 1925

    return sents, labels

class PersonalityDataset(Dataset):
    def __init__(self,tokenizer,data_type,data_dir,device,max_len,class_num,batch_size=16,first_cluster=False):
        self.data_path = f"data/{data_dir}/train.txt"
        self.data_dir = data_dir
        self.device = device
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.first_cluster = first_cluster
        self.bert_tokenizer = AutoTokenizer.from_pretrained(BERT_URL)

        self.sentence_embedding_name = f"train_data_embedding/{data_dir}/{data_dir}_content_embedding.pth"
        if not os.path.exists(self.sentence_embedding_name):
            self.bert = AutoModel.from_pretrained(BERT_URL)
            self.bert.to(device)
            self.device = device
            for parameter in self.bert.base_model.parameters():
                parameter.requires_grad = False

        self.data_dir = data_dir
        self.data_type = data_type
        self.target_length = max_len
        self.batch_size = batch_size
        self.content_embeddings = None

        if data_dir == "Essays":
            self.cluster_num = 50
        elif data_dir == "Kaggle":
            self.cluster_num = 30
        self.class_num = class_num

        if data_dir == "Essays":
            self.prompt = "A personal essay that shows author's personality traits of "
        elif data_dir == "Kaggle":
            self.prompt = "Some personal sentences that show traits of "
        self.sep = ", "

        self.inputs = [] # 带prompt的输入
        self.targets = [] # 标签列表[[1,0,0,1,0],...]
        self.target_seq = [] # 与self.inputs相同内容
        self.bert_inputs = [] # 单纯的训练文本
        self.clusters = []

        if data_dir == "Essays":
            labels = ['extraversion','neuroticism','agreeableness','conscientiousness','openness']
            self.idx2label = {a:b for a,b in enumerate(labels)}
        elif data_dir == "Kaggle":
            labels = ['introversion','extraversion','intuition','sensing','feeling','thinking','perception','judgment']
            self.idx2label = {a:b for a,b in enumerate(labels)}

        self._build_examples()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        source_ids = self.inputs[idx]["input_ids"].squeeze()
        target_ids = self.target_seq[idx]['input_ids'].squeeze()
        target = torch.tensor(self.targets[idx])
        clusters = torch.tensor(self.clusters[idx])
        bert_ids = self.bert_inputs[idx]["input_ids"].squeeze()

        src_mask = self.inputs[idx]["attention_mask"].squeeze()
        bert_mask = self.bert_inputs[idx]["attention_mask"].squeeze()
        return {"source_ids": source_ids, "source_mask": src_mask, "labels": target, "target_ids": target_ids,
                "bert_ids": bert_ids, "bert_mask": bert_mask, "clusters": clusters}

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def kaggle_data_processing(self,text):
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

    def _build_examples(self):
        # inputs, targets = get_transformed_data(self.data_path, self.data_dir, self.data_type)
        inputs, targets = get_data(self.data_path, self.data_type)

        print("Begin clustering ... ")
        if not os.path.exists(self.sentence_embedding_name):
            print("未查询到训练数据的嵌入化特征，开始提取训练数据的特征...")
            sentence_embeddings = []
            for examples in tqdm(inputs,desc="Embedding..."):
                examples = self.kaggle_data_processing(examples)
                example = examples.split('|||')[:30]
                encoded_input = self.bert_tokenizer(example, padding='max_length', truncation=True, max_length=self.max_len,
                                                    return_tensors='pt').to(self.device)
                num_sentences = encoded_input["input_ids"].size(0)
                if num_sentences < 30:
                    pad = 30 - num_sentences

                    encoded_input["input_ids"] = torch.cat(
                        [encoded_input['input_ids'], torch.zeros((pad, self.max_len),dtype=torch.long,device=self.device)], dim=0
                    )
                    encoded_input["attention_mask"] = torch.cat(
                        [encoded_input['attention_mask'], torch.zeros((pad, self.max_len),dtype=torch.long,device=self.device)], dim=0
                    )
                with torch.no_grad():
                    model_output = self.bert(encoded_input["input_ids"], attention_mask=encoded_input["attention_mask"])

                sentence_embedding = self.mean_pooling(model_output, encoded_input['attention_mask'])
                sentence_embeddings.append(sentence_embedding.cpu())
            sentence_embeddings = torch.cat(sentence_embeddings)
            print(sentence_embeddings.shape)
            torch.save(sentence_embeddings, self.sentence_embedding_name)
            print("训练数据嵌入化完毕，保存成功！")
        else:
            print("查询到训练数据嵌入向量，加载中...")
            sentence_embeddings = torch.load(self.sentence_embedding_name)
            print("加载成功！")

        km = KMeans(n_clusters=self.cluster_num, random_state=SEED, n_init="auto").fit(sentence_embeddings.numpy())
        sen2cluster = {sen: cluster for sen, cluster in zip(inputs, km.labels_)}
        cluster_centers = km.cluster_centers_

        self.content_embeddings = torch.from_numpy(cluster_centers)

        targets = [str(t) for t in targets]
        input2target = {b: a for a, b in zip(targets, inputs)}
        label_set = list(sorted(set(targets)))
        temp_inputs, temp_labels = [], []

        label_same_dict = {l: [] for l in label_set}
        for sent in inputs:
            label_same_dict[input2target[sent]].append(sent)

        pad_seq = [self.tokenizer.pad_token] * 1
        pad_seq = " ".join(pad_seq)
        self.pad_seq = pad_seq
        pad_label = [0] * self.class_num
        """
        label_same_dict = {
            [0,0,0,0,1]: [[text1],[text2],...],
            ...
        } 
        """
        for sent_label, sent_list in label_same_dict.items():
            if len(sent_list) % self.batch_size == 0:
                temp_inputs.append(sent_list)
                temp_labels.append([sent_label] * len(sent_list))
            else:
                left_num = len(sent_list) % self.batch_size
                origin_len = len(sent_list)
                sent_list += [pad_seq] * (self.batch_size - left_num)
                temp_inputs.append(sent_list)
                append_labels = [sent_label] * origin_len + [str(pad_label)] * (self.batch_size - left_num)
                temp_labels.append(append_labels)

        temp_inputs = sum(temp_inputs, [])
        temp_labels = sum(temp_labels, [])
        self.targets = [eval(a) for a in temp_labels]

        for s, label in zip(temp_inputs, self.targets):
            label_words = self.sep.join([self.idx2label[idx] for idx, l in enumerate(label) if l == 1])
            input = self.prompt + label_words + " : " + s + " " + self.tokenizer.eos_token if label_words != "" else pad_seq
            target = self.prompt + label_words + " : " + s + " " + self.tokenizer.eos_token if label_words != "" else pad_seq

            bert_input = s if label_words != "" else self.bert_tokenizer.pad_token

            tokenized_input = self.tokenizer.batch_encode_plus(
                [input], max_length=self.max_len, truncation=True, padding='max_length',
                return_tensors="pt"
            )
            tokenized_target = self.tokenizer.batch_encode_plus(
                [target], max_length=self.max_len, truncation=True, padding='max_length',
                return_tensors="pt"
            )

            self.inputs.append(tokenized_input)
            self.target_seq.append(tokenized_target)
            self.bert_inputs.append(
                self.bert_tokenizer(bert_input, add_special_tokens=True, max_length=self.max_len, truncation=True,
                                    padding='max_length', return_tensors="pt"))
            if s in sen2cluster.keys():
                self.clusters.append(sen2cluster[s])
            else:
                self.clusters.append(-1)
