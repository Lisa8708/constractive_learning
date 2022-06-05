# simcse task_unsupervised_train.py
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
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

jieba.initialize()

# 基本参数
model_type = 'SimBERT'
pooling = 'cls'
dropout_rate = 0.1 #默认情况下，同一个batch内，不同样本的dropout是不一样的，相当于：x*np.random.binomial(1, p=1-p, size=x.shape)/(1-p)
maxlen = 64
batch_size = 64  # 对内存要求高
epochs = 3

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

# 加载数据集
datasets = {
    '%s-%s' % (task_name, f):
    load_data('%s/%s/%s.%s' % (data_path, task_name, task_name, f))
    for f in ['train.data', 'test.data']
}
# 单条数据格式：(文本1, 文本2, 标签)

# 建立分词器
tokenizer = get_tokenizer(dict_path)

# 建立模型
encoder = get_encoder(
    config_path,
    checkpoint_path,
    pooling=pooling,
    dropout_rate=dropout_rate
)

# 语料id化
all_names, all_weights, all_token_ids, all_labels = [], [], [], []
train_token_ids = []
# name: 'train', 'valid', 'test'
for name, data in datasets.items():
    a_token_ids, b_token_ids, labels = convert_to_ids(data, tokenizer, maxlen)
    all_names.append(name)
    all_weights.append(len(data))
    all_token_ids.append((a_token_ids, b_token_ids))
    all_labels.append(labels)
    train_token_ids.extend(a_token_ids)  # 合并
    train_token_ids.extend(b_token_ids)

# 构造训练集
np.random.shuffle(train_token_ids)
train_token_ids = train_token_ids[:10000]
# 以train+valid+test全体为测试集，train+valid+test中随机挑1万条作为训练集。数据变多了loss会变得很小:
# simcse的训练目标本身跟句子相似度任务没有直接联系，训练数据越多，只能让simcse本身的训练loss越小，不能保证句子相似度任务有提升。

print("train:", len(train_token_ids), len(train_token_ids[0])) # 1089296=2*len(train+valid+test) 64

class data_generator(DataGenerator):
    """训练语料生成器
    """
    def __iter__(self, random=False):
        batch_token_ids = []
        for is_end, token_ids in self.sample(random):
            batch_token_ids.append(token_ids)  ##
            batch_token_ids.append(token_ids)
#             print("test:", len(token_ids), len(batch_token_ids))
            if len(batch_token_ids) == self.batch_size * 2 or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)  # 将多个序列截断或补齐为相同长度
                batch_segment_ids = np.zeros_like(batch_token_ids)
                batch_labels = np.zeros_like(batch_token_ids[:, :1])  ## 生成负样本
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids = []


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


print(" ************************* 训练 *******************************")
# SimCSE训练
encoder.summary()
encoder.compile(loss=simcse_loss, optimizer=Adam(1e-5))

train_generator = data_generator(train_token_ids, batch_size)  #
print("train_generator:", len(train_generator)) # 11607 训练数据29万
encoder.fit(train_generator.forfit(),
    steps_per_epoch=len(train_generator),
    epochs=epochs)
# 每一轮epoch需要执行多少steps，也就是多少steps，才能认为一轮epoch结束。
# 那么衍生问题就是，一个step是怎么度量？其实就是规定每个step加载多少数据，也就是batch_size。
# 他们的关系如下：steps_per_epoch=len(x_train)/batch_size
# 一句话概括，就是对于整个训练数据集，generator要在多少步内完成一轮遍历（epoch），
# 从而也就规定了每步要加载多少数据（batch_size）。
encoder.save_weights(model_save_path)

print(" ************************* 测试模型效果 *******************************")
# 语料向量化
all_vecs = []  # names=3*segment=2
for a_token_ids, b_token_ids in all_token_ids:
    a_vecs = encoder.predict([a_token_ids, np.zeros_like(a_token_ids)], verbose=True) # len(test) * 768
    b_vecs = encoder.predict([b_token_ids, np.zeros_like(b_token_ids)], verbose=True)
    all_vecs.append((a_vecs, b_vecs))

# 标准化，相似度，相关系数
all_corrcoefs = []
for (a_vecs, b_vecs), labels in zip(all_vecs, all_labels):
    a_vecs = l2_normalize(a_vecs)
    b_vecs = l2_normalize(b_vecs)
    sims = (a_vecs * b_vecs).sum(axis=1)  # train: len(train) prob
#     print("sims:", sims)
    corrcoef = compute_corrcoef(labels, sims)
    all_corrcoefs.append(corrcoef)

all_corrcoefs.extend([
    np.average(all_corrcoefs),
    np.average(all_corrcoefs, weights=all_weights)  # all_weights 数据量
])

for name, corrcoef in zip(all_names + ['avg', 'w-avg'], all_corrcoefs):
    print('%s: %s' % (name, corrcoef))

# <------------------------ 预测结果输出 ----------------->
# 加载数据集
print(" ************************* 预测 *******************************")
pred_data = load_data(f"{data_path}/{task_name}/{task_name}.test.data")
pred_out_path = f"{data_path}/{task_name}/predicts/test.simcse.txt"
pred_data = load_data(pred_data)
a_token_ids, b_token_ids, labels = convert_to_ids(pred_data, tokenizer, maxlen)
a_vecs = encoder.predict([a_token_ids, np.zeros_like(a_token_ids)], verbose=True) # len(test) * 768
b_vecs = encoder.predict([b_token_ids, np.zeros_like(b_token_ids)], verbose=True)
a_vecs = l2_normalize(a_vecs)
b_vecs = l2_normalize(b_vecs)
sims = (a_vecs * b_vecs).sum(axis=1)  # train: len(train) prob

# 预测集的效果输出：
with open(pred_out_path, 'w') as wf:
    for i in sims:
        wf.write(str(i) + '\n')