# 基于BERT的模型剪枝

## 1 数据集

将[GAIIC-2021](https://tianchi.aliyun.com/competition/entrance/531852/information?lang=zh-cn)的带标签线下训练集进行随机划分，得到训练集与测试集，分别含有8000和2000条数据，实现多标签、17分类任务。数据集路径：`datasets/{train/test}.csv`。

## 2 实验环境

```
python == 3.9
torch == 1.7.1
transformers == 4.20.1
sentencepiece == 0.1.96
protobuf == 3.20.0
textpruner == 1.1.post1
thop == 0.0.31.post2005241907
```

## 3 流程

TextPruner库支持Vocabulary Pruning和Transformer Pruning。由于采用Train From Scratch，即：使用给定数据集上的文本语料进行预训练，并不像已经开源的BERT权重一样，在大规模语料库上训练，并在下游小任务上微调。因此，词典里的token都会在下游任务上见到，所以并不需要进行Vocabulary Pruning，仅采用Transformer Pruning即可。

1、基于训练集和测试集的文本语料，进行BERT预训练；

2、基于带标签训练集，对BERT进行下游文本分类任务的微调；

3、使用textpruner库，导入微调后的含全连接层的模型进行剪枝，生成压缩后的BERT；

4、根据压缩后的BERT以及新生成的配置文件，重新进行一次下游任务微调，最后得到压缩后的在文本分类任务上的模型。

## 4 实验结果

评估指标：1 - mlogloss；预训练轮数：100；微调轮数：10。

|   Experiment    | target_ffn_size | target_num_of_heads | n_iters | head_even_masking | use_logits | Metric     | FLOPs (G) | Params (M) |
| :-------------: | :-------------: | :-----------------: | :-----: | :---------------: | :--------: | :--------: |:---------:|:----------:|
|    Baseline     | -               | -                   | -       | -                 | -          | 0.8990     | 21.74    | 102.3     |
|   Pruning - 1   | 1536            | 6                   | 1       | True              | False      | 0.9175     | 10.87    | 59.8      |
|   Pruning - 2   | 1536            | 6                   | 16      | False             | True       | 0.9193     | 10.92    | 60.0      |
| **Pruning - 3** | **768**         | **4**               | **16**  | **False**         | **True**   | **0.9114** | **6.14** | **41.3**  |
|   Pruning - 4   | 384             | 1                   | 16      | False             | True       | 0.8930     | 2.72     | 27.9      |

Q: 压缩后的模型怎么分类效果比不剪枝的模型还好？

A: 可能说明了这一点：原来的BERT在这个任务上是过拟合的，而剪枝恰好能缓解过拟合，也印证了CV上的模型压缩方法[SFP](https://arxiv.org/abs/1808.06866)得到的结论。

## 5 运行

**生成语料库、词典**：

```python
python3 corpus_vocab.py
```

**预训练**：

```python
python3 pretrain.py
```

BERT和配置文件生成在`./pretrained_bert`目录下。

**微调**：

```python
python3 train_val.py
```

model的路径：`./trained_models/model.pth`。

**在测试集上推理**：

```python
python3 evaluate.py --evaluate-model-path ./trained_models/model.pth --pretrained-bert-dir ./pretrained_bert/
```

**分析计算量、参数量**：

```python
python3 flops_params.py --pretrained-bert-dir ./pretrained_bert/
```

**剪枝**：

```python
python3 prune.py --prune-bert-save-dir ./pruned_bert/$PRUNED_BERT_DIR$
```

`./pruned_bert/$PRUNED_BERT_DIR$/{PRUNING_CONFIG}`目录下，存放了剪枝后的BERT以及新的配置文件。

**再微调**：

```python
python3 train_val.py --pretrained-bert-dir ./pruned_bert/$PRUNED_BERT_DIR$/$PRUNING_CONFIG$ --train-model-save-dir ./pruned_models/$PRUNED_BERT_DIR$/$PRUNING_CONFIG$
```

剪枝后微调过的model路径：`./pruned_models/$PRUNED_BERT_DIR$/$PRUNING_CONFIG$/model.pth`。

**在测试集上推理**：

```python
python3 evaluate.py --evaluate-model-path ./pruned_models/$PRUNED_BERT_DIR$/$PRUNING_CONFIG$/model.pth --pretrained-bert-dir ./pruned_bert/$PRUNED_BERT_DIR$/$PRUNING_CONFIG$
```

**分析计算量、参数量**：

```python
python3 flops_params.py --pretrained-bert-dir ./pruned_bert/$PRUNED_BERT_DIR}/{PRUNING_CONFIG}
```

## 参考链接

[TextPruner调参](https://zhuanlan.zhihu.com/p/469103382)

[demo](https://blog.51cto.com/u_14156307/5274012)

