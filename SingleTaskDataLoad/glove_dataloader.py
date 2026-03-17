import pandas as pd
import torch
from torch.utils.data import Dataset

def read_data(file_url):
    data = pd.read_csv(
        file_url,
        header=0,
        encoding="utf-8",
        encoding_errors="ignore",
    )
    return data['text'].values, data[['EXT','NEU','AGR','CON','OPN']].values

class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

def collate_fn(batch, vocab, MAX_SEQUENCE_LENGTH):
    texts, labels = zip(*batch)
    batch_ids = []
    lengths = []
    for text in texts:
        tokens = text.split()
        # 分词并转换为id
        ids = [vocab.get(w.lower(), vocab["<UNK>"]) for w in tokens]
        seq_len = min(len(ids), MAX_SEQUENCE_LENGTH)
        lengths.append(seq_len)
        # 截断或补齐
        if len(ids) > MAX_SEQUENCE_LENGTH:
            ids = ids[:MAX_SEQUENCE_LENGTH]
        else:
            ids.extend([vocab["<PAD>"]] * (MAX_SEQUENCE_LENGTH - len(ids)))
        batch_ids.append(ids)
    batch_ids = torch.tensor(batch_ids, dtype=torch.long)
    lengths = torch.tensor(lengths, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.float32)
    return batch_ids, lengths, labels