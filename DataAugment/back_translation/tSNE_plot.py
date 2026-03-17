import pandas as pd
import numpy as np
import torch
from torch import cosine_similarity
from transformers import AutoTokenizer, AutoModel
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm import tqdm

from SingleTextAff import SingleTextAffectiveSpace


def func(tokenizer,jina):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    url = "/hy-tmp/models"
    # tokenizer = AutoTokenizer.from_pretrained("/hy-tmp/models/jina-embeddings-v2-base-en")
    # jina = AutoModel.from_pretrained('/hy-tmp/models/jina-embeddings-v2-base-en', trust_remote_code=True).to(device)
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
    affective_space = {row[0]: row[1:] for _, row in affective_xlsx.iterrows()}

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
        inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = jina(**inputs)
            cls_emb = outputs.last_hidden_state[:, 0, :]

        return cls_emb

    i = 0

    def augment_sacbt_bert(df, bt_texts, sim_thres=0.90, affect_thres=0.15):
        # print("评估回译样本质量")
        augmented_rows = []
        i = []
        for index, row in df.iterrows():
            torch.cuda.empty_cache()
            text = row['text']
            labels = [row['EXT'], row['NEU'], row['AGR'], row['CON'], row['OPN']]

            bt_text = bt_texts[index]

            # 1️⃣ BERT语义相似度
            emb1 = get_jina_embedding(text)
            emb2 = get_jina_embedding(bt_text)
            sim = cosine_similarity(emb1, emb2, dim=1)
            #
            # 2️⃣ AffectSpace特征距离
            f1 = get_affective_vector(text)
            f2 = get_affective_vector(bt_text)
            affect_dist = np.linalg.norm(f1 - f2)

            # 3️⃣ 双约束筛选
            if sim > sim_thres and affect_dist < affect_thres:
                i.append(index)
            # else:
            #     augmented_rows.append([999] + labels)
            print(f"{index+1}/{len(df)}...")
        return np.array(i)

    data = pd.read_csv(
        "/hy-tmp/dataset/Essays/原始数据/essays_revise.csv",
        header=0,
        encoding="utf-8",
        encoding_errors="ignore",
    )
    bt_data = pd.read_csv(
        "/hy-tmp/dataset/Essays/回译/法德回译无筛选.csv",
        header=0,
        encoding="utf-8",
        encoding_errors="ignore",
    )
    bt_texts = bt_data['text']
    i = augment_sacbt_bert(data, bt_texts)
    return i

# ----------------------------
# 1. 设备设置
# ----------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# ----------------------------
# 2. 加载 Jina Embeddings 模型和分词器
# ----------------------------
model_name = "/hy-tmp/models/bert-base-uncased"
# model_name = "/hy-tmp/models/bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
model.to(device)
model.eval()  # 推理模式

extractor = AffectiveFeatureExtractor()
# Jina embeddings 使用 mean pooling
def mean_pooling(token_embeddings, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


# ----------------------------
# 3. 文本嵌入函数（支持 batch + GPU）
# ----------------------------
def encode_texts(texts, batch_size=8, max_length=1024):
    all_embeddings = []
    texts = texts.tolist()
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding texts"):
        batch = texts[i:i + batch_size]
        # Tokenize
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
        # Move to GPU
        encoded = {k: v.to(device) for k, v in encoded.items()}
        psycholinguistic = extractor.extract_features_batch(batch)
        psycholinguistic = torch.from_numpy(psycholinguistic).to(device)
        with torch.no_grad():
            outputs = model(**encoded)
            # embeddings = mean_pooling(outputs.last_hidden_state, encoded['attention_mask'])
            embeddings = outputs.last_hidden_state[:,0,:]

            # Normalize embeddings (Jina recommends L2 norm)
            # embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            features = torch.cat((embeddings,psycholinguistic), dim=1)
            all_embeddings.append(features.cpu().numpy())

    return np.vstack(all_embeddings)


# ----------------------------
# 4. 加载数据
# ----------------------------
def load_text_column(csv_path):
    df = pd.read_csv(csv_path, header=0, encoding="utf-8", encoding_errors="ignore")
    return df.iloc[:, 0].fillna("").astype(str)


source_texts = load_text_column("/hy-tmp/dataset/Essays/原始数据/essays_revise.csv")
translated_texts = load_text_column("/hy-tmp/dataset/Essays/回译/法德回译有筛选.csv")

print(f"Loaded {len(source_texts)} source and {len(translated_texts)} translated texts.")

# 可选：对齐长度（防止不一致）
min_len = min(len(source_texts), len(translated_texts))
idx = func(tokenizer,model)
print("筛选后长度:",len(idx))
source_texts = source_texts[idx]
translated_texts = translated_texts[idx]

# ----------------------------
# 5. 向量化（GPU加速）
# ----------------------------
print("Encoding source texts...")
source_embeddings = encode_texts(source_texts, batch_size=8, max_length=512)

print("Encoding translated texts...")
translated_embeddings = encode_texts(translated_texts, batch_size=8, max_length=512)

# ----------------------------
# 6. t-SNE 降维
# ----------------------------
all_embeddings = np.vstack([source_embeddings, translated_embeddings])
labels = ['Source'] * len(source_embeddings) + ['Translated'] * len(translated_embeddings)

print("Running t-SNE...")
tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42, verbose=1)
embedding_2d = tsne.fit_transform(all_embeddings)

# ----------------------------
# 7. 绘图
# ----------------------------
plt.figure(figsize=(12, 9))
colors = {'Source': 'tab:blue', 'Translated': 'tab:red'}
markers = {'Source': 's', 'Translated': '^'}
sizes = {'Source': 70, 'Translated': 55}

for label in ['Translated', 'Source']:
    idx = [i for i, l in enumerate(labels) if l == label]
    x, y = embedding_2d[idx].T
    plt.scatter(x, y, c=colors[label], marker=markers[label], s=sizes[label],
                edgecolors='black', linewidth=0.5, alpha=0.7, label=label)

plt.legend(title="Text Type", fontsize=11, title_fontsize=11)
plt.title("t-SNE Visualization of Source vs Back-Translated Texts (Bert Embeddings)", fontsize=14)
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.gca().spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.savefig("./3.pdf",format="pdf")
plt.show()