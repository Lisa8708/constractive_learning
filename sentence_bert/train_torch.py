# 基于sentence_bert 进行fine-tune训练
# 参考文档：https://wmathor.com/index.php/archives/1497/
from sentence_transformers import SentenceTransformer, SentencesDataset, InputExample, losses, evaluation, util
import torch
from torch.utils.data import DataLoader
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from tqdm import tqdm
import math
import numpy as np
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import warnings
warnings.filterwarnings('ignore')
device = torch.device("cpu")

# train_examples = [InputExample(texts=['My first sentence', 'My second sentence'], label=0.8),
#                    InputExample(texts=['Another pair', 'Unrelated sentence'], label=0.3)]

# 初始参数
batch_size = 64
epochs = 3
evaluation_steps = 1000

# data path train=290634,test=36135
data_path = "../data"
task_name = 'BQ'
train_file_path = f"{data_path}/{task_name}/{task_name}.train.data"
test_file_path = f"{data_path}/{task_name}/{task_name}.valid.data"
pred_file_path = f"{data_path}/{task_name}/{task_name}.test.data"
pred_out_path = pred_file_path + '.pred.sbert'

# model_path
model_base_path = '../../pretrained_model/bert-base-uncased/'
# sentence-transformer-bert-base-nli-mean-tokens, bert-base-nli-mean-tokens, bert-base-nli-cls-token
model_save_path = f"{data_path}/{task_name}/model/sbert"

# 获取数据
def loadData(path):
    res = []
    with open(path, 'r') as rf:
        lines = rf.readlines()
        for line in lines:
            text_a, text_b, label = line.strip().split('\t')
            res.append(InputExample(texts=[text_a, text_b], label=float(label)))
    return res

train_datas = loadData(train_file_path)
test_datas = loadData(test_file_path)

print("data loaded...", "train={},test={}".format(len(train_datas), len(test_datas)))

# 加载预训练模型 cls mean max https://public.ukp.informatik.tu-darmstadt.de/reimers/sentence-transformers/v0.2/
model = SentenceTransformer(model_base_path, device='cpu')  # cuda

# 训练集
# np.random.shuffle(train_datas)
# train_datas = train_datas[:10000]
train_dataset = SentencesDataset(train_datas, model)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)  # batch_size 每个batch加载batch_size个样本
# 模型损失函数：余弦相似度损失
train_loss = losses.CosineSimilarityLoss(model)

# 设定warmup
warmup_steps = math.ceil(len(train_dataset) * epochs / batch_size * 0.1) #10% of train data for warm-up
# 使用Warmup预热学习率的方式,即先用最初的小学习率训练，然后每个step增大一点点，直到达到最初设置的比较大的学习率时（注：此时预热学习率完成），采用最初设置的学习率进行训练（注：预热学习率完成后的训练过程，学习率是衰减的），有助于使模型收敛速度变快，效果更佳。

# 效果评估
evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_datas, name='test_data')

# Tune the model
model.fit(train_objectives=[(train_dataloader, train_loss)],
          epochs=epochs,
          warmup_steps=warmup_steps, # 进行warm up过程的步数。
          evaluator=evaluator,
          evaluation_steps=evaluation_steps,
          output_path=model_save_path)


# 加载训练好的模型
# model = SentenceTransformer(model_save_path,device='cuda')

# 计算相似度
def getSim(model, text_a, text_b):
#     print("vec:", model.encode(text_a))
    return util.pytorch_cos_sim(model.encode(text_a), model.encode(text_b))[0][0]

# 效果评估
with open(pred_file_path, 'r') as rf, open(pred_out_path, 'w') as wf:
    lines = rf.readlines()
    for line in tqdm(lines):
        text_a, text_b, label = line.split("\t")
        sim = getSim(model, text_a, text_b) # 点乘计算两个句子的相似度
#         print(line.strip('\n') + '\t' + str(sim.numpy()))
        wf.write(line.strip('\n') + '\t' + str(sim.numpy()) + "\n")