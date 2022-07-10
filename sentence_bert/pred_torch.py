# 基于sentence_bert 进行fine-tune训练
# 参考文档：https://wmathor.com/index.php/archives/1497/
from sentence_transformers import SentenceTransformer, SentencesDataset, InputExample, losses, evaluation, util
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import warnings
warnings.filterwarnings('ignore')

# 初始参数
batch_size = 64
epochs = 10
evaluation_steps = 1000

# data
data_path = "../data"
task_name = 'BQ'
pred_file_path = f"{data_path}/{task_name}/{task_name}.test.data"
pred_out_path = f"{data_path}/{task_name}/pred/test.sbert.txt"

# 加载训练好的模型
# model_base_path = '../../pretrained_model/sentence-transformer-bert-base-nli-mean-tokens'
model_base_path = f"{data_path}/{task_name}/model/sbert"

model = SentenceTransformer(model_base_path, device='cuda')

# 计算相似度
def getSim(model, text_a, text_b):
    return util.pytorch_cos_sim(model.encode(text_a), model.encode(text_b))[0][0]

# 数据预测
with open(pred_file_path, 'r') as rf, open(pred_out_path, 'w') as wf:
    lines = rf.readlines()
    for line in tqdm(lines):
        text_a, text_b, label = line.split("\t")
        sim = getSim(model, text_a, text_b) # 点乘计算两个句子的相似度
#         print(line.strip('\n') + '\t' + str(sim.numpy()))
        wf.write(line.strip('\n') + '\t' + str(sim.numpy()) + "\n")