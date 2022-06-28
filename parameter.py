# -*- coding: UTF-8 -*-
import os
import argparse
import multiprocessing
import torch


def build_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    parser.add_argument('--dataset-path', type=str, default='./datasets')

    parser.add_argument('--pretrained-bert-dir', type=str, default='./pretrained_bert')

    parser.add_argument('--bert-config-path', type=str, default='./helper/bert_config.json')

    parser.add_argument('--batch-size', type=int,
                        help='batch size', default=16)

    parser.add_argument('--num-workers', type=int,
                        help='number of workers for data loader', default=multiprocessing.cpu_count())

    parser.add_argument('--num-pretrain-epochs', type=int, help='number for pretraining', default=100)

    parser.add_argument('--epochs', type=int, help='number for training', default=10)

    parser.add_argument('--n-splits', type=int, default=5)

    parser.add_argument('--seq-len', type=int, default=100)

    parser.add_argument('--train-model-save-dir', type=str,
                        help='train model save directory', default='./trained_models')

    parser.add_argument('--prune-bert-save-dir', type=str,
                        help='pruned BERT save directory', default='./pruned_bert')

    parser.add_argument('--to-be-pruned-path', type=str, default='./trained_models/model.pth')

    parser.add_argument('--to-be-analyzed-path', type=str, default='./pruned_models/model.pth')

    parser.add_argument('--evaluate-model-path', type=str, default='./trained_models/model.pth')

    return parser


def get_parameter():
    parser = build_parser()
    args = parser.parse_args()

    args.train_set_path = os.path.join(args.dataset_path, 'train.csv')

    args.test_set_path = os.path.join(args.dataset_path, 'test.csv')

    args.corpus_path = os.path.join(args.pretrained_bert_dir, 'corpus.txt')

    args.vocab_path = os.path.join(args.pretrained_bert_dir, 'vocab.txt')

    return args


if __name__ == '__main__':
    get_parameter()
