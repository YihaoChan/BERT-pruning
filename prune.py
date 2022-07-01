# -*- coding: UTF-8 -*-
from transformers import BertTokenizer
from transformers import BertForSequenceClassification
from textpruner import summary, TransformerPruner, TransformerPruningConfig, GeneralConfig
from torch.utils.data import DataLoader
from helper.dataset import TextDataset
from evaluate import test_pro
import numpy as np
from parameter import get_parameter

args = get_parameter()

model = BertForSequenceClassification.from_pretrained(args.pretrained_bert_dir, num_labels=17,
                                                      problem_type='multi_label_classification')

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

"""
importance_score: Loss对weight的偏导 * weight
                  (weight.grad * weight).view(weight.size(0), n_heads, -1).sum(dim=(0, 2)).abs()

head_even_masking: 每一层mask掉不同数量的head，一共有12个layer，每个layer有12个head
    importance[12, 12]: 12层里面，每一层12个head，按层进行重要性打分；
    importance_order[144]: 对【所有的head】进行重要性得分的排序，得到[144]里面重要性的从低到高的【下标】；
                           这些下标对应的head，分布在【不同层】；
    mask[12, 12]: 根据排序的144个重要性得分，根据不同层的head的重要性排序，将target_size个heads在[12, 12]中进行mask置0。
    
    因为排序的时候是全部一起排，所以mask指定数量的head之后，每一层的head数量就基本不同了。
    至此，就实现了在每一层mask掉不同数量的head。
"""
pruner = TransformerPruner(model, transformer_pruning_config=transformer_pruning_config, general_config=general_config)

pruner.prune(dataloader=test_loader, save_model=True)

tokenizer.save_pretrained(save_directory=pruner.save_dir)

print(summary(model))