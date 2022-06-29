# -*- coding: UTF-8 -*-
from transformers import BertTokenizer
from transformers import BertForSequenceClassification
from textpruner import summary, TransformerPruner, TransformerPruningConfig, GeneralConfig
from torch.utils.data import DataLoader
from helper.dataset import TextDataset
from evaluate import test_pro
import numpy as np
import torch
from parameter import get_parameter

args = get_parameter()

model = BertForSequenceClassification.from_pretrained(args.pretrained_bert_dir, num_labels=17,
                                                      problem_type='multi_label_classification')

model.load_state_dict(torch.load(args.to_be_pruned_path))

tokenizer = BertTokenizer.from_pretrained(args.pretrained_bert_dir)

test_df = test_pro(args)

test_dataset = TextDataset(test_df, np.arange(test_df.shape[0]))

test_loader = DataLoader(
    test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
)

general_config = GeneralConfig(use_device='auto', output_dir=args.prune_bert_save_dir)

transformer_pruning_config = TransformerPruningConfig(
    target_ffn_size=384, target_num_of_heads=1,
    pruning_method='iterative', n_iters=16,
    head_even_masking=False, use_logits=True)

pruner = TransformerPruner(model, transformer_pruning_config=transformer_pruning_config, general_config=general_config)

pruner.prune(dataloader=test_loader, save_model=True)

tokenizer.save_pretrained(save_directory=pruner.save_dir)

print(summary(model))
