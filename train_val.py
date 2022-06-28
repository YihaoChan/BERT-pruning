# -*- coding: UTF-8 -*-
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification
from helper.fgm_adv import FGM
from helper.seed import seed
import torch
import torch.nn as nn
import numpy as np
import os
import warnings
import pandas as pd
from tqdm import tqdm
from helper.dataset import TextDataset
from parameter import get_parameter

warnings.filterwarnings("ignore")

torch.cuda.empty_cache()


def train_pro(args):
    train_df = pd.read_csv(args.train_set_path, header=None)
    train_df.columns = ['report_ID', 'description', 'region']

    train_df.drop(['report_ID'], axis=1, inplace=True)
    train_df['description'] = [i.strip('|').strip() for i in train_df['description'].values]
    train_df['region'] = [i.strip('|').strip() for i in train_df['region'].values]

    train_num = len(train_df)

    for train_idx in tqdm(range(train_num)):
        des = train_df.loc[train_idx, 'description']

        des = [int(word) for word in des.split(' ')]

        train_df.loc[train_idx, 'description'] = des

    return train_df


def train(args):
    train_df = train_pro(args)

    model = BertForSequenceClassification.from_pretrained(args.pretrained_bert_dir, num_labels=17,
                                                          problem_type='multi_label_classification')

    model = model.to(args.device)

    train_dataset = TextDataset(train_df, np.arange(train_df.shape[0]))

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )

    iters = len(train_loader)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=5e-4)

    criterion = torch.nn.BCEWithLogitsLoss()

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=3, T_mult=2, last_epoch=-1)

    for epoch in range(args.epochs):
        epoch += 1

        model.train(True)

        fgm = FGM(model)

        for batch_idx, batch in enumerate(train_loader):
            batch_idx += 1

            data = batch['input_ids']
            label = batch['labels']

            data = data.type(torch.LongTensor).to(args.device)
            label = label.type(torch.FloatTensor).to(args.device)

            output = model(data)[0].to(args.device)

            loss = criterion(output, label)

            optimizer.zero_grad()

            # 正常的grad
            loss.backward(retain_graph=True)

            # 对抗训练
            fgm.attack()
            loss_adv = criterion(output, label)
            loss_adv.backward(retain_graph=True)
            fgm.restore()

            # 梯度下降，更新参数
            optimizer.step()
            model.zero_grad()
            lr_scheduler.step()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)

            if batch_idx % 20 == 0:
                print('epoch: {}, batch: {} / {}, loss: {:.3f}'.format(epoch, batch_idx, iters, loss.item()))

        torch.save(model.state_dict(), os.path.join(args.train_model_save_dir, 'model.pth'))


def main():
    args = get_parameter()

    seed()

    if not os.path.exists(args.train_model_save_dir):
        os.makedirs(args.train_model_save_dir)

    train(args)


if __name__ == '__main__':
    main()
