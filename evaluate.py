# -*- coding: UTF-8 -*-
from torch.utils.data import DataLoader
from helper.dataset import TextDataset
from transformers import BertForSequenceClassification
from helper.metric import metric
from helper.seed import seed
import torch
import numpy as np
import warnings
import pandas as pd
from tqdm import tqdm
from parameter import get_parameter

warnings.filterwarnings("ignore")

torch.cuda.empty_cache()


def test_pro(args):
    test_df = pd.read_csv(args.test_set_path, header=None)
    test_df.columns = ['report_ID', 'description', 'region']

    test_df.drop(['report_ID'], axis=1, inplace=True)
    test_df['description'] = [i.strip('|').strip() for i in test_df['description'].values]
    test_df['region'] = [i.strip('|').strip() for i in test_df['region'].values]

    train_num = len(test_df)

    for train_idx in tqdm(range(train_num)):
        des = test_df.loc[train_idx, 'description']

        des = [int(word) for word in des.split(' ')]

        test_df.loc[train_idx, 'description'] = des

    return test_df


def load_model(args, net, weight_path):
    model = net.to(args.device)

    model.load_state_dict(torch.load(weight_path))

    model.eval()

    return model


@torch.no_grad()
def pred(args, model, test_loader):
    model.eval()

    pred_list = []
    label_list = []

    for batch in test_loader:
        data = batch['input_ids']
        label = batch['labels']

        data = data.type(torch.LongTensor).to(args.device)
        label = label.type(torch.FloatTensor).to(args.device)

        output = model(data)[0].to(args.device)

        pred_list += output.sigmoid().detach().cpu().numpy().tolist()

        label_list += label.detach().cpu().numpy().tolist()

    metric_loss = metric(label_list, pred_list)

    return metric_loss


def main():
    args = get_parameter()

    seed()

    model = BertForSequenceClassification.from_pretrained(args.pretrained_bert_dir, num_labels=17,
                                                          problem_type='multi_label_classification')

    test_df = test_pro(args)

    model = load_model(args, model, args.evaluate_model_path)

    test_dataset = TextDataset(test_df, np.arange(test_df.shape[0]))

    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )

    loss = pred(args, model, test_loader)

    print("Test mlogloss:", np.mean(loss))


if __name__ == '__main__':
    main()
