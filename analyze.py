# -*- coding: UTF-8 -*-
from transformers import BertForSequenceClassification
from textpruner import summary, inference_time
import torch
from parameter import get_parameter

args = get_parameter()

model = BertForSequenceClassification.from_pretrained(args.pretrained_bert_dir, num_labels=17,
                                                      problem_type='multi_label_classification')

model.load_state_dict(torch.load(args.to_be_analyzed_path))

print(summary(model))

dummy_inputs = [torch.randint(low=0, high=10000, size=(4, 128))]
print("Inference time:")
inference_time(model.to(args.device), dummy_inputs)
