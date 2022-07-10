# # simcse task_unsupervised_pred.py
#! -*- coding: utf-8 -*-
# SimCSE 中文测试

from task_utils import *
import sys
import tensorflow as tf
from bert4keras.optimizers import Adam
from bert4keras.snippets import DataGenerator, sequence_padding
import jieba
import os
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

jieba.initialize()

model_type = 'SimBERT'
pooling = 'cls'
dropout_rate = 0.1 # 默认情况下，同一个batch内，不同样本的dropout是不一样的，相当于：x*np.random.binomial(1, p=1-p, size=x.shape)/(1-p)
maxlen = 64
batch_size = 64
epochs = 10
steps_per_epoch = 1000

# bert配置
base_path = '../../pretrained_model/chinese_simbert_L-4_H-312_A-12/'
config_path = base_path + 'bert_config.json'
checkpoint_path = base_path + 'bert_model.ckpt'
dict_path = base_path + 'vocab.txt'

# data path
data_path = '../data'
task_name = 'BQ'

# model save
model_save_path = f"{data_path}/{task_name}/models/simcse.weights"

def simcse_loss(y_true, y_pred):
    """用于SimCSE训练的loss
    """
    # 构造标签
    idxs = K.arange(0, K.shape(y_pred)[0])
    idxs_1 = idxs[None, :]
    idxs_2 = (idxs + 1 - idxs % 2 * 2)[:, None]
    y_true = K.equal(idxs_1, idxs_2)
    y_true = K.cast(y_true, K.floatx())
    # 计算相似度
    y_pred = K.l2_normalize(y_pred, axis=1)
    similarities = K.dot(y_pred, K.transpose(y_pred))
    similarities = similarities - tf.eye(K.shape(y_pred)[0]) * 1e12
    similarities = similarities * 20  # ?? *20
# （2）)loss计算的时候，同一句话放两次，那计算得到的similarities矩阵对角线就是自身和自身，那这个样本对的loss是怎么理解，看代码是减去了一个1e12，后面又乘以了个20?
# 你认真看(1)式，有个τ，原论文也有，默认是τ=0.05，相当于乘以20。不scale是不行的，因为cos的范围是-1～1，直接softmax的话loss会降不下去。
    loss = K.categorical_crossentropy(y_true, similarities, from_logits=True)
    return K.mean(loss)

# 建立分词器
tokenizer = get_tokenizer(dict_path)

# 建立模型
encoder = get_encoder(
    config_path,
    checkpoint_path,
    pooling=pooling,
    dropout_rate=dropout_rate
)
# SimCSE训练
encoder.compile(loss=simcse_loss, optimizer=Adam(1e-5))

# load model
encoder.load_weights(model_save_path)
encoder.summary()

# 加载数据集
pred_data = load_data(f"{data_path}/{task_name}/{task_name}.test.data")
a_token_ids, b_token_ids, labels = convert_to_ids(pred_data, tokenizer, maxlen)
a_vecs = encoder.predict([a_token_ids, np.zeros_like(a_token_ids)], verbose=True) # len(test) * 768
b_vecs = encoder.predict([b_token_ids, np.zeros_like(b_token_ids)], verbose=True)
a_vecs = l2_normalize(a_vecs)
print("a_vecs:", a_vecs[0])
b_vecs = l2_normalize(b_vecs)
print("b_vecs:", b_vecs[0])
sims = (a_vecs * b_vecs).sum(axis=1)  # train: len(train) prob
print("sims:", sims)

# 预测集的效果输出：
pred_out_path = f"{data_path}/{task_name}/predicts/test.simcse.txt"
with open(pred_out_path, 'w') as wf:
    for i in sims:
        wf.write(str(i) + '\n')
