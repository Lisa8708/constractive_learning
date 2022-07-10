#! -*- coding: utf-8 -*-
# 测试SimBERT训练后的效果，利用文本相似度进行测试  使用训练后的weights
# 训练环境：tensorflow 1.14 + keras 2.3.1 + bert4keras 0.7.7

from io import TextIOBase
import numpy as np
from collections import Counter
from bert4keras.backend import keras, K
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer, load_vocab
from bert4keras.snippets import sequence_padding
# from bert4keras.snippets import uniout, open
# from bert4keras.snippets import open
from keras.models import Model
# from common_data_process import *
from bert4keras.layers import Loss
import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

maxlen = 64

# bert small
base_path = '../../pretrained_model/chinese_simbert_L-4_H-312_A-12/'
config_path = base_path + 'bert_config.json'
checkpoint_path = base_path + 'bert_model.ckpt'
dict_path = base_path + 'vocab.txt'

class TotalLoss(Loss):
    """loss分两部分，一是seq2seq的交叉熵，二是相似度的交叉熵。
    """
    def compute_loss(self, inputs, mask=None):
        loss1 = self.compute_loss_of_seq2seq(inputs, mask)
        loss2 = self.compute_loss_of_similarity(inputs, mask)
        self.add_metric(loss1, name='seq2seq_loss')
        self.add_metric(loss2, name='similarity_loss')
        return loss1 + loss2

    def compute_loss_of_seq2seq(self, inputs, mask=None):
        y_true, y_mask, _, y_pred = inputs
        y_true = y_true[:, 1:]  # 目标token_ids
        y_mask = y_mask[:, 1:]  # segment_ids，刚好指示了要预测的部分
        y_pred = y_pred[:, :-1]  # 预测序列，错开一位
        loss = K.sparse_categorical_crossentropy(y_true, y_pred)
        loss = K.sum(loss * y_mask) / K.sum(y_mask)
        return loss

    def compute_loss_of_similarity(self, inputs, mask=None):
        _, _, y_pred, _ = inputs
        y_true = self.get_labels_of_similarity(y_pred)  # 构建标签
        y_pred = K.l2_normalize(y_pred, axis=1)  # 句向量归一化
        similarities = K.dot(y_pred, K.transpose(y_pred))  # 相似度矩阵
        similarities = similarities - K.eye(K.shape(y_pred)[0]) * 1e12  # 排除对角线
        similarities = similarities * 30  # scale
        loss = K.categorical_crossentropy(
            y_true, similarities, from_logits=True
        )
        return loss

    def get_labels_of_similarity(self, y_pred):
        idxs = K.arange(0, K.shape(y_pred)[0])
        idxs_1 = idxs[None, :]
        idxs_2 = (idxs + 1 - idxs % 2 * 2)[:, None]
        labels = K.equal(idxs_1, idxs_2)
        labels = K.cast(labels, K.floatx())
        return labels

# 加载并精简词表，建立分词器
token_dict, keep_tokens = load_vocab(
    dict_path=dict_path,
    simplified=True,
    startswith=['[PAD]', '[UNK]', '[CLS]', '[SEP]'],
)
tokenizer = Tokenizer(token_dict, do_lower_case=True)

# 建立加载模型
bert = build_transformer_model(
    config_path,
    checkpoint_path,
    with_pool='linear',
    application='unilm',
    keep_tokens=keep_tokens,  # 只保留keep_tokens中的字，精简原字表
    return_keras_model=False,
)

encoder = keras.models.Model(bert.model.inputs, bert.model.outputs[0])
seq2seq = keras.models.Model(bert.model.inputs, bert.model.outputs[1])

outputs = TotalLoss([2, 3])(bert.model.inputs + bert.model.outputs)
model = keras.models.Model(bert.model.inputs, outputs)

# 加载模型
# model.load_weights('../../data/XNLI-1.0/simbert/best_model.weights')

# 测试相似度效果
a_token_ids = []

def cal_vec(text):
    token_ids, segment_ids = tokenizer.encode(text)
    # 获取CLS向量
    vec = encoder.predict([[token_ids], [segment_ids]])[0]
#     vec = model.predict([[token_ids], [segment_ids]])[0][0]
    vec /= (vec**2).sum()**0.5  ## 归一化  vec = vec/sqrt(sum(vec**2))
    return vec

with open("../../data/XNLI-1.0/test.csv", "r") as rf, open('../../data/XNLI-1.0/test.pred.simbert.txt','w') as wf:
    texts = rf.readlines()
    all_count = len(texts)
    right_count = 0
    for text in texts:
        text_a, text_b, label = text.split("\t")
        sim = np.dot(cal_vec(text_a), cal_vec(text_b)) # 点乘计算两个句子的相似度
        wf.write(text.strip('\n') + '\t' + str(sim) + "\n")
        pred_label = 1 if sim > 0.5 else 0
        if int(label) == pred_label:
            right_count += 1
    print(f"acc: {round(right_count/all_count, 4)}")