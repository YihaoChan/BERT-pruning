import os
import pandas as pd
from tqdm import tqdm
from parameter import get_parameter


def dataset_pro(dataset_path):
    df = pd.read_csv(dataset_path, header=None)

    df.columns = ['report_ID', 'description', 'region']

    df.drop(['report_ID'], axis=1, inplace=True)
    df['description'] = [i.strip('|').strip() for i in df['description'].values]

    data_num = len(df)

    for idx in tqdm(range(data_num)):
        des = df.loc[idx, 'description']

        des = [int(word) for word in des.split(' ')]

        df.loc[idx, 'description'] = des

    return df


def gen_corpus(args):
    def write_corpus(corpus_file, *dfs):
        print('Total number of corpus file components: ' + str(len(dfs)))

        for df in dfs:
            for idx in range(len(df)):
                des = df.iloc[idx, 0]
                des = ' '.join(str(item) for item in des)
                # Text should be one-sentence-per-line, with empty lines between documents.
                corpus_file.write(str(des) + '\n\n')

    train_df = dataset_pro(args.train_set_path)
    test_df = dataset_pro(args.test_set_path)

    with open(args.corpus_path, 'a') as f:
        f.seek(0)
        f.truncate()

        write_corpus(f, train_df, test_df)

        f.close()


def gen_vocab(args):
    def write_vocab(vocab_list, *deses):
        print('Total number of vocab file components: ' + str(len(deses)))

        for des in deses:
            for sentence in des:
                for word_train in sentence:
                    if word_train not in vocab_list:
                        vocab_list.append(word_train)

    train_des = dataset_pro(args.train_set_path).iloc[:, 0].tolist()
    test_des = dataset_pro(args.test_set_path).iloc[:, 0].tolist()

    vocab = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']

    write_vocab(vocab, train_des, test_des)

    with open(args.vocab_path, 'a') as f:
        f.seek(0)
        f.truncate()

        for i in tqdm(range(len(vocab))):
            f.write(str(vocab[i]) + '\n')


def main():
    args = get_parameter()

    if not os.path.exists(args.pretrained_bert_dir):
        os.makedirs(args.pretrained_bert_dir)

    gen_corpus(args)
    gen_vocab(args)


if __name__ == '__main__':
    main()
