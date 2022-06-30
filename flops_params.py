# -*- coding: utf-8 -*-
import torch
from transformers import BertForSequenceClassification, BertTokenizer
from thop import profile
from parameter import get_parameter
from textpruner import summary, inference_time
import warnings

warnings.filterwarnings("ignore")


def _input_constructor(args, num_labels, input_shape, tokenizer):
    max_length = input_shape[1]

    # sequence for subsequent flops calculation
    model_input_ids = []
    model_attention_mask = []
    model_token_type_ids = []
    for _ in range(input_shape[0]):
        inp_seq = ""
        inputs = tokenizer.encode_plus(
            inp_seq,
            add_special_tokens=True,
            truncation_strategy='longest_first',
        )
        # print(inputs)

        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
        attention_mask = [1] * len(input_ids)
        padding_length = max_length - len(input_ids)
        pad_token = tokenizer.pad_token_id
        pad_token_segment_id = 0
        input_ids = input_ids + ([pad_token] * padding_length)
        attention_mask = attention_mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
        assert len(input_ids) == max_length
        assert len(attention_mask) == max_length
        assert len(token_type_ids) == max_length
        model_input_ids.append(input_ids)
        model_attention_mask.append(attention_mask)
        model_token_type_ids.append(token_type_ids)

    labels = torch.randn(size=(1, num_labels)).to(args.device)

    # Batch size input_shape[0], sequence length input_shape[128]
    inputs = {
        "input_ids": torch.tensor(model_input_ids).to(args.device),
        "token_type_ids": torch.tensor(model_token_type_ids).to(args.device),
        "attention_mask": torch.tensor(model_attention_mask).to(args.device),
    }

    inputs.update({"labels": labels})
    # print([(k, v.size()) for k, v in inputs.items()])
    return inputs


def cal_plm_flops(args, num_labels, model, tokenizer, max_seq_length):
    inputs = _input_constructor(args, num_labels, (1, max_seq_length), tokenizer)

    inputs_for_flops = (
        inputs.get("input_ids", None),
        inputs.get("attention_mask", None),
        inputs.get("token_type_ids", None),
        inputs.get("position_ids", None),
        inputs.get("head_mask", None),
        inputs.get("input_embeds", None),
        inputs.get("labels", None),
    )

    total_ops, _ = profile(model, inputs=inputs_for_flops, verbose=False)

    print("******FLOPs: %.2fG******" % (2 * total_ops / (1000 ** 3)))  # MACs * 2 = FLOPs


def cal_plm_params(model):
    print(summary(model))

    dummy_inputs = [torch.randint(low=0, high=10000, size=(4, 128))]
    print("Inference time:")
    inference_time(model, dummy_inputs)


def main():
    args = get_parameter()

    num_labels = 17

    tokenizer = BertTokenizer.from_pretrained(args.pretrained_bert_dir)
    model = BertForSequenceClassification.from_pretrained(args.pretrained_bert_dir, num_labels=num_labels,
                                                          problem_type='multi_label_classification')

    model = model.to(args.device)

    max_seq_length = 128

    cal_plm_params(model)

    cal_plm_flops(args, num_labels, model, tokenizer, max_seq_length)


if __name__ == '__main__':
    main()
