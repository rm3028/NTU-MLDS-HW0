
from enum import Enum, auto
import torch
from torch.utils.data import Dataset
import pandas as pd
from transformers import AutoTokenizer

class DatasetType(Enum):
    TrainingLabel = auto()
    TrainingUnLabel = auto()
    TestingData = auto()


class HW0Dataset(Dataset):
    def __init__(self, dataset_dir, datasetType = DatasetType.TrainingLabel):
        csv_separator = ' \+\+\+\$\+\+\+ '

        self.dataset_dir = dataset_dir
        self.datasetType = datasetType

        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

        self.training_label_df = pd.read_csv(dataset_dir + '/training_label.csv', names=['label', 'text'], sep=csv_separator, engine='python')
        self.training_nolabel_df = pd.read_csv(dataset_dir + '/training_nolabel.csv', names=['text'], sep=csv_separator, engine='python')
        self.testing_data_df = pd.read_csv(dataset_dir + '/testing_data.csv', sep=r'(?<=\d|\w),(?!\s)', engine='python')

        self.training_label_df['encoded_text'] = self.training_label_df['text'].apply(self.tokenizer.encode, add_special_tokens=True)
        self.training_nolabel_df['encoded_text'] = self.training_nolabel_df['text'].apply(self.tokenizer.encode, add_special_tokens=True)
        self.testing_data_df['encoded_text'] = self.testing_data_df['text'].apply(self.tokenizer.encode, add_special_tokens=True)

        self.training_label_ts = torch.tensor(self.training_label_df['label'].values)

        dataset_tss = []

        for encoded_text in self.training_label_df['encoded_text']:
            dataset_tss.append(torch.LongTensor(encoded_text))
        for encoded_text in self.training_nolabel_df['encoded_text']:
            dataset_tss.append(torch.LongTensor(encoded_text))
        for encoded_text in self.testing_data_df['encoded_text']:
            dataset_tss.append(torch.LongTensor(encoded_text))

        dataset_ts = torch.nn.utils.rnn.pad_sequence(dataset_tss, batch_first=True)

        self.training_data_ts = dataset_ts[0:len(self.training_label_df)]
        self.training_nolabel_ts = dataset_ts[len(self.training_label_df):len(self.training_label_df)+len(self.training_nolabel_df)]
        self.testing_data_ts = dataset_ts[len(self.training_label_df)+len(self.training_nolabel_df):]

        self.max_seq_len = torch.max(dataset_ts).tolist() + 1

    def __len__(self):
        if self.datasetType == DatasetType.TrainingLabel:
            return len(self.training_data_ts)
        elif self.datasetType == DatasetType.TrainingUnLabel:
            return len(self.training_nolabel_ts)
        elif self.datasetType == DatasetType.TestingData:
            return len(self.testing_data_ts)
        else:
            return 0

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.datasetType == DatasetType.TrainingLabel:
            return { 'text': self.training_data_ts[idx], 'label': self.training_label_ts[idx] }
        elif self.datasetType == DatasetType.TrainingUnLabel:
            return {'text': self.training_nolabel_ts[idx]}
        elif self.datasetType == DatasetType.TestingData:
            return {'text': self.testing_data_ts[idx]}
        else:
            return None

