import gc

import torch
import spacy
from torch import cosine_similarity
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM
from SingleTextAff import SingleTextAffectiveSpace

import numpy as np
import pandas as pd

# ========== 初始化 BERT ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("/hy-tmp/models/bert-base-uncased")
emb_model = AutoModel.from_pretrained("/hy-tmp/models/bert-base-uncased").to(device)

fr_tokenizer = AutoTokenizer.from_pretrained("/hy-tmp/models/opus-mt-en-fr")
fr_model = AutoModelForSeq2SeqLM.from_pretrained("/hy-tmp/models/opus-mt-en-fr").to(device)
fr_model.eval()

de_tokenizer = AutoTokenizer.from_pretrained("/hy-tmp/models/opus-mt-fr-de")
de_model = AutoModelForSeq2SeqLM.from_pretrained("/hy-tmp/models/opus-mt-fr-de").to(device)
de_model.eval()

en_tokenizer = AutoTokenizer.from_pretrained("/hy-tmp/models/opus-mt-de-en")
en_model = AutoModelForSeq2SeqLM.from_pretrained("/hy-tmp/models/opus-mt-de-en").to(device)
en_model.eval()

extractor = SingleTextAffectiveSpace()

nlp = spacy.load("en_core_web_sm")


def get_affective_vector(text):
    """平均AffectSpace向量"""
    # print("获取SenticNet8特质")
    sentic_feat = extractor.extract_features_batch(text)
    return sentic_feat

def get_bert_embedding(text):
    # print("获取Bert嵌入向量")
    """获取BERT [CLS] 向量"""
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512, padding="max_length").to(device)
    with torch.no_grad():
        outputs = emb_model(**inputs)
        cls_emb = outputs.last_hidden_state[:,0,:]
    return cls_emb

def back_translate(long_text, fr_tok, fr_mod, de_tok, de_mod, en_tok, en_mod):
    # print("回译中...")
    doc = nlp(long_text)
    sentences = [sent.text.strip() for sent in doc.sents]

    # en->fr
    fr_target_inputs = fr_tok(sentences, return_tensors="pt", padding=True, truncation=True, max_length=64).to(device)
    with torch.no_grad():
        fr_translated_output = fr_mod.generate(
            **fr_target_inputs,
            max_new_tokens=64,
            do_sample=True,
            temperature=0.85,
            top_k=50,
            top_p=0.92,
            num_return_sequences=1,
            pad_token_id=fr_tokenizer.pad_token_id,
            eos_token_id=fr_tokenizer.eos_token_id,
            early_stopping=True
        )
    fr_translated_text = fr_tok.batch_decode(fr_translated_output, skip_special_tokens=True)

    # fr->de
    de_target_inputs = de_tok(fr_translated_text, return_tensors="pt", padding=True, truncation=True,
                              max_length=64).to(device)
    with torch.no_grad():
        de_translated_output = de_mod.generate(
            **de_target_inputs,
            max_new_tokens=64,
            do_sample=True,
            temperature=0.85,
            top_k=50,
            top_p=0.92,
            num_return_sequences=1,
            pad_token_id=de_tokenizer.pad_token_id,
            eos_token_id=de_tokenizer.eos_token_id,
            early_stopping=True
        )
    de_translated_text = de_tok.batch_decode(de_translated_output, skip_special_tokens=True)

    # de->en
    en_target_inputs = en_tok(de_translated_text, return_tensors="pt", padding=True, truncation=True,
                              max_length=64).to(device)
    with torch.no_grad():
        en_translated_output = en_mod.generate(
            **en_target_inputs,
            max_new_tokens=64,
            do_sample=True,
            temperature=0.85,
            top_k=50,
            top_p=0.92,
            num_return_sequences=1,
            pad_token_id=en_tokenizer.pad_token_id,
            eos_token_id=en_tokenizer.eos_token_id,
            early_stopping=True
        )
    en_translated_text = en_tok.batch_decode(en_translated_output, skip_special_tokens=True)

    # print("回译完成！！！")
    return " ".join(en_translated_text)

def augment_sacbt_bert(df, sim_thres=0.60, affect_thres=0.20):
    # 评估回译样本质量
    augmented_rows = []
    for index, row in df.iterrows():
        text = row['posts']
        labels = [row['I/E'],row['N/S'],row['F/T'],row['P/J']]

        bt_text = back_translate(text,fr_tokenizer,fr_model,de_tokenizer,de_model,en_tokenizer,en_model)

        #  BERT语义相似度
        emb1 = get_bert_embedding(text)
        emb2 = get_bert_embedding(bt_text)
        sim = cosine_similarity(emb1, emb2)

        #  AffectSpace特征距离
        f1 = get_affective_vector(text)
        f2 = get_affective_vector(bt_text)
        affect_dist = np.linalg.norm(f1 - f2)

        #  双约束筛选
        if sim > sim_thres and affect_dist < affect_thres:
            augmented_rows.append([bt_text] + labels)

        print(f"{index+1}/{len(df)}...")
    return augmented_rows

def read_data():
    data = pd.read_csv(
        "/hy-tmp/dataset/MBTI/back_translation/mbti_original.csv",
        header=0,
        encoding="utf-8",
        encoding_errors="ignore",
    )
    return data

if __name__ == '__main__':
    columns = ['posts', 'I/E', 'N/S', 'F/T', 'P/J']
    data = read_data()

    augmented_data = pd.DataFrame(augment_sacbt_bert(data),columns=columns)

    augmented_data.to_csv('/hy-tmp/dataset/MBTI/back_translation/augment_data.csv', index=False,encoding='utf-8')



