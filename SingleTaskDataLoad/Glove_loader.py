import numpy as np
import torch

EMBEDDING_DIM = 300
MAX_SEQUENCE_LENGTH = 1024
# 加载glove词典进来
def load_GloVe(fileurl,embedding_dim=EMBEDDING_DIM):
    word_to_vec = {}
    with open(fileurl,encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vec = np.array(values[1:], dtype=np.float32)
            if len(vec) != embedding_dim:
                continue
            word_to_vec[word] = vec
    return word_to_vec

# 构建词汇表和嵌入矩阵
def build_vocab_and_embedding_matrix(texts,word_to_vec,embedding_dim=EMBEDDING_DIM,min_freq=1):
    word_freq = {}
    for text in texts:
        for word in text.split():
            word_freq[word] = word_freq.get(word,0) + 1
    vocab = {"<PAD>":0, "<UNK>":1}
    idx = 2
    for word, freq in word_freq.items():
        # 高于最低频率且单词在GloVe中存在，才会存入词汇表
        if freq >= min_freq and word in word_to_vec:
            # 往词汇表vocab里加新的词汇
            vocab[word] = idx
            idx += 1
    vocab_size = len(vocab)

    # 构建嵌入矩阵
    embedding_matrix = np.random.normal(loc=0, scale=1, size=(vocab_size, embedding_dim)).astype(np.float32)
    embedding_matrix[0] = 0.0
    count = 0
    for word, idx in vocab.items():
        if word in word_to_vec:
            embedding_matrix[idx] = word_to_vec[word]
        else:
            # 对pad和unk搞一个嵌入向量 都用随机初始化的嵌入向量
            embedding_matrix[idx] = embedding_matrix[1]
    return vocab, embedding_matrix
