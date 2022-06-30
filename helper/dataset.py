# -*- coding: UTF-8 -*-
from torch.utils.data import Dataset
from parameter import get_parameter
import torch

class TextDataset(Dataset):
    def __init__(self, df, idx):
        super().__init__()
        self.args = get_parameter()

        self.df = df.loc[idx, :].reset_index(drop=True)

        self.description = df['description'].values

        self.labels = df['region'].values

    @staticmethod
    def get_dummy(classes):
        """
        标签转为0/1向量
        """
        label = [0] * 17

        if classes == '':
            return label
        else:
            temp = [int(i) for i in classes.strip().split(' ')]

            for i in temp:
                label[i] = 1

        return label

    def des_padding(self, des_list):
        """
        截断文本，少的用858填充，多的直接截断
        """
        des_len = len(des_list)

        if des_len > self.args.seq_len:
            des = des_list[:self.args.seq_len]
        else:
            des = des_list + [858] * (self.args.seq_len - des_len)

        return des

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        des = self.description[idx]

        label = self.labels[idx]

        padding_des = self.des_padding(des)

        label = self.get_dummy(label)

        return {'input_ids': torch.tensor(padding_des, dtype=torch.long), 'labels': torch.tensor(label, dtype=torch.long)}
