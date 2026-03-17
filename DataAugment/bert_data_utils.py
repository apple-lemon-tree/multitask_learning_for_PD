from torch.utils.data import Dataset
import pandas as pd
import torch

from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.classes.preprocessor import TextPreProcessor
from tqdm import tqdm



def get_data_and_label(data_dir, data_type):
    df = pd.read_csv(f'data/{data_dir}/{data_type}.txt', sep="\t", header=None)
    data = df.iloc[:, 1].values.tolist()
    labels = df.iloc[:, 0].values.tolist()

    return data, labels


def get_data(data_dir):

    train_data, train_label = get_data_and_label(data_dir, "train")
    train_label = [eval(label) for label in train_label]

    return train_data, train_label


class ClassificationDataset(Dataset):
    def __init__(self, tokenizer, data_type, data_dir, max_len=128, train_filter=False):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data_dir = data_dir
        self.data_type = data_type
        self.target_length = max_len
        self.train_filter = train_filter

        self.inputs = []
        self.targets = None

        self._build_examples()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target = torch.tensor(self.targets[index])

        src_mask = self.inputs[index]["attention_mask"].squeeze()

        return {"source_ids": source_ids, "source_mask": src_mask, "labels": target}

    def _build_examples(self):
        inputs, self.targets = get_data(self.data_dir, self.data_type, self.train_filter)

        for i in tqdm(range(len(inputs))):
            input = inputs[i].strip()

            tokenized_input = self.tokenizer(
                input, max_length=self.max_len, padding='max_length', truncation=True,
                return_tensors="pt",
            )

            self.inputs.append(tokenized_input)
