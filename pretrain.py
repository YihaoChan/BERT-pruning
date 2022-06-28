# coding:utf-8
import os
import warnings
from transformers import BertConfig, BertForMaskedLM, BertTokenizer, LineByLineTextDataset, \
    DataCollatorForLanguageModeling, \
    Trainer, TrainingArguments
from helper.seed import seed
from parameter import get_parameter

warnings.filterwarnings('ignore')


def main():
    args = get_parameter()

    if not os.path.exists(args.pretrained_bert_dir):
        os.makedirs(args.pretrained_bert_dir)

    seed()

    config = BertConfig.from_pretrained(args.bert_config_path)

    model = BertForMaskedLM(config=config)

    tokenizer = BertTokenizer.from_pretrained(args.vocab_path)

    dataset = LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path=args.vocab_path,
        block_size=args.seq_len
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    training_args = TrainingArguments(
        output_dir=args.pretrained_bert_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.num_pretrain_epochs,
        per_device_train_batch_size=args.batch_size,
        save_steps=10000,
        save_total_limit=2
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset
    )

    trainer.train()

    trainer.save_model(args.pretrained_bert_dir)


if __name__ == '__main__':
    main()
