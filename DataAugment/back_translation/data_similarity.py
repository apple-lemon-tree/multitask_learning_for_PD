import numpy as np
import pandas as pd
import torch
from torch import cosine_similarity

from transformers import AutoModel, AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("/hy-tmp/models/bert-base-uncased")
jina = AutoModel.from_pretrained('/hy-tmp/models/bert-base-uncased',trust_remote_code=True).to(device)
jina.eval()

float_cols_dtype = {i: 'float32' for i in range(1, 101)}
dtype_mapping = {
    0: 'str',
    **float_cols_dtype
}
affective_xlsx = pd.read_csv(
    filepath_or_buffer="/hy-tmp/dataset/affectivespace/affective.csv",
    header=None,
    encoding='utf-8',
    dtype=dtype_mapping,
    low_memory=False,
)
affective_space = {row[0]:row[1:] for _, row in affective_xlsx.iterrows()}

def get_affective_vector(text):
    """平均AffectSpace向量"""
    # print("获取SenticNet8特质")
    tokens = text.lower().split()
    vectors = [np.array(affective_space[t], dtype=float)
               for t in tokens if t in affective_space]
    if len(vectors) == 0:
        return np.zeros(100)
    return np.mean(vectors, axis=0)

def get_jina_embedding(text):
    # print("获取Jina嵌入向量")
    """获取BERT [CLS] 向量"""
    inputs = tokenizer(text,add_special_tokens=True, return_tensors='pt', truncation=True, max_length=512,padding='max_length').to(device)
    with torch.no_grad():
        outputs = jina(**inputs)
        cls_emb = outputs[0][:,0,:]
    return cls_emb



def augment_sacbt_bert(df, bt_data,sim_thres=0.90, affect_thres=0.15):
    # print("评估回译样本质量")
    augmented_rows = []
    i = 0
    for index, row in df.iterrows():
        torch.cuda.empty_cache()
        text = row['text']
        labels = [row['EXT'],row['NEU'],row['AGR'],row['CON'],row['OPN']]

        bt_text = bt_data['text'][index]

        # BERT语义相似度
        emb1 = get_jina_embedding(text)
        emb2 = get_jina_embedding(bt_text)
        # sim = cosine_similarity([emb1], [emb2])[0][0]
        sim = cosine_similarity(emb1, emb2,dim=1)
        #
        # AffectSpace特征距离
        f1 = get_affective_vector(text)
        f2 = get_affective_vector(bt_text)
        affect_dist = np.linalg.norm(f1 - f2)

        # 3 双约束筛选
        if sim > sim_thres and affect_dist < affect_thres:
            i= i + 1
        else:
            bt_data['text'][index] = 'low equality'
        print(f"{index+1}/{len(df)}...")
    bt_data.to_csv("/hy-tmp/dataset/Essays/en2fr2de2en_bt_essays2.csv", index=False,encoding='utf-8')
    print("i:",i)

if __name__ == '__main__':
    data = pd.read_csv(
        "/hy-tmp/dataset/Essays/essays_revise.csv",
        header=0,
        encoding="utf-8",
        encoding_errors="ignore",
    )
    bt_data = pd.read_csv(
        "/hy-tmp/dataset/Essays/en2fr2de2en_bt_essays.csv",
        header=0,
        encoding="utf-8",
        encoding_errors="ignore",
    )
    augment_sacbt_bert(data,bt_data)