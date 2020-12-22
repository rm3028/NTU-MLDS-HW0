
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
    def __init__(self, dataset_dir, max_seq_length=-1, datasetType=DatasetType.TrainingLabel):
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
        self.testing_id_ts = torch.tensor(self.testing_data_df['id'].values)

        self.max_seq_length = max_seq_length

        training_label_tss = []
        for encoded_text in self.training_label_df['encoded_text']:
            training_label_tss.append(torch.LongTensor(
                encoded_text if self.max_seq_length < 0 or len(encoded_text) <= self.max_seq_length else encoded_text[0:self.max_seq_length]))
        self.training_data_ts = torch.nn.utils.rnn.pad_sequence(training_label_tss, batch_first=True)

        training_nolabel_tss = []
        for encoded_text in self.training_nolabel_df['encoded_text']:
            training_nolabel_tss.append(torch.LongTensor(
                encoded_text if self.max_seq_length < 0 or len(encoded_text) <= self.max_seq_length else encoded_text[0:self.max_seq_length]))
        self.training_nolabel_ts = torch.nn.utils.rnn.pad_sequence(training_nolabel_tss, batch_first=True)

        testing_data_tss = []
        for encoded_text in self.testing_data_df['encoded_text']:
            testing_data_tss.append(torch.LongTensor(
                encoded_text if self.max_seq_length < 0 or len(encoded_text) <= self.max_seq_length else encoded_text[0:self.max_seq_length]))
        self.testing_data_ts = torch.nn.utils.rnn.pad_sequence(testing_data_tss, batch_first=True)

        self.vocab_size = max(torch.max(self.training_data_ts).item(), torch.max(self.training_nolabel_ts).item(), torch.max(self.testing_data_ts).item()) + 1
        self.label_num = torch.max(self.training_label_ts).item() + 1

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
            return {'text': self.training_data_ts[idx], 'label': self.training_label_ts[idx]}
        elif self.datasetType == DatasetType.TrainingUnLabel:
            return {'text': self.training_nolabel_ts[idx]}
        elif self.datasetType == DatasetType.TestingData:
            return {'text': self.testing_data_ts[idx], 'id': self.testing_id_ts[idx]}
        else:
            return None

