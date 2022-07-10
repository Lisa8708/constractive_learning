# https://github.com/zhoujx4/NLP-Series-sentence-embeddings
import argparse
import os
import json
import numpy as np
import scipy.stats
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import BertTokenizer, BertModel, BertConfig

os.environ["CUDA_VISIBLE_DEVICES"] = '3'
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# parser
parser = argparse.ArgumentParser()
# parser.add_argument('--model_path', type=str, required=True)
parser.add_argument('--save_path', type=str, default="../data/kefu_text/model/SimCSE/supervised.kg_14d_3w.weights")
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--learning_rate', type=float, default=4e-5)
parser.add_argument('--maxlen', type=int, default=64)
parser.add_argument('--early_stop', type=int, default=5)
args = parser.parse_args()

# pretrained model
model_path = '../../pretrained_model/bert-base-uncased/'
tokenizer = BertTokenizer.from_pretrained(model_path + 'vocab.txt')
Config = BertConfig.from_pretrained(model_path)

# file path 
train_file_path = "../data/kefu_text/kg_14d.json.simcse.supervised"
test_file_path = "../data/kefu_text/fy_dev_50w.csv"
pred_file_path = "../data/kefu_text/testSenRes2"

# load data
def load_train_vocab(path):
    data = []
    with open(path) as f:
        for i in f:
            data.append(json.loads(i))
    return data

def load_test_data(path):
    data = []
    with open(path) as f:
        for i in f:
            d = i.split("\t")
            sentence1 = d[0]
            sentence2 = d[1]
            score = d[2]
            data.append([sentence1,sentence2,score])
    return data

train_data = load_train_vocab(train_file_path)
np.random.shuffle(train_data)
train_data = train_data[:30000]
test_data = load_test_data(test_file_path)
np.random.shuffle(test_data)
test_data = test_data[:5000]
pred_data = load_test_data(pred_file_path)

print(" ... load data ... train={}, test={}, pred={}".format(len(train_data), len(test_data), len(pred_data)))

class model(nn.Module):
    def __init__(self, model_path):
        super(model, self).__init__()
        self.bert = BertModel.from_pretrained(model_path, config=Config)

    def forward(self, input_ids, attention_mask, token_type_ids):
        x1 = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        output = x1.last_hidden_state[:, 0]
        return output


class TrainDataset(Dataset):
    def __init__(self, data, tokenizer, maxlen, transform=None, target_transform=None):
        self.data = data
        self.tokenizer = tokenizer
        self.maxlen = maxlen
        self.transform = transform
        self.target_transform = target_transform

    def text_to_id(self, source):
        origin = source['origin']
        entailment = source['entailment']
        contradiction = source['contradiction']
        sample = self.tokenizer([origin, entailment, contradiction],
                                max_length=self.maxlen,
                                truncation=True,
                                padding='max_length',
                                return_tensors='pt')
        return sample

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.text_to_id(self.data[idx])

class TestDataset:
    def __init__(self, data, tokenizer, maxlen):
        self.tokenizer = tokenizer
        self.maxlen = maxlen
        self.traget_idxs = self.text_to_id([x[0] for x in data])
        self.source_idxs = self.text_to_id([x[1] for x in data])
        self.label_list = [int(x[2]) for x in data]
        assert len(self.traget_idxs['input_ids']) == len(self.source_idxs['input_ids'])

    def text_to_id(self, source):
        sample = self.tokenizer(source, max_length=self.maxlen, truncation=True, padding='max_length',
                                return_tensors='pt')
        return sample

    def get_data(self):
        return self.traget_idxs, self.source_idxs, self.label_list

def compute_loss(y_pred, lamda=0.05):
    row = torch.arange(0, y_pred.shape[0], 3, device='cuda')
    col = torch.arange(y_pred.shape[0], device='cuda')
    col = torch.where(col % 3 != 0)[0].cuda()
    y_true = torch.arange(0, len(col), 2, device='cuda')
    similarities = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=2)
    # torch自带的快速计算相似度矩阵的方法
    similarities = torch.index_select(similarities, 0, row)
    similarities = torch.index_select(similarities, 1, col)
    # 屏蔽对角矩阵即自身相等的loss
    similarities = similarities / lamda
    # 论文中除以 temperature 超参 0.05
    loss = F.cross_entropy(similarities, y_true)
    return torch.mean(loss)

def train(dataloader, testdata, model, optimizer):
    model.train()
    size = len(dataloader.dataset)
    best_corrcoef = 0
    early_stop = 0
    for epoch in range(4):
        for batch, data in tqdm(enumerate(dataloader), total=len(dataloader), desc="Epoch:{}".format(epoch)):
            input_ids = data['input_ids'].view(len(data['input_ids']) * 3, -1).to(device)
            attention_mask = data['attention_mask'].view(len(data['attention_mask']) * 3, -1).to(device)
            token_type_ids = data['token_type_ids'].view(len(data['token_type_ids']) * 3, -1).to(device)
            pred = model(input_ids, attention_mask, token_type_ids)
            loss = compute_loss(pred)
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if batch % 200 == 0:
                loss, current = loss.item(), batch * int(len(input_ids) / 3)
                # print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
                model.eval()  # 意味着显示关闭丢弃dropout
                corrcoef = test(testdata, model)
                model.train()
                print("\nSpearman's correlation：", format(corrcoef))
                if corrcoef > best_corrcoef:
                    early_stop = 0
                    best_corrcoef = corrcoef
                    torch.save(model.state_dict(), args.save_path)
                    print("Update the best spearman's correlation：{}，saving model！".format(corrcoef))
                else:
                    early_stop += 1
                    if early_stop > args.early_stop:
                        print("\nEarly stop!The best spearman's correlation is {}".format(best_corrcoef))
                        break

def cal_sim(traget_idxs, source_idxs, model):
    with torch.no_grad():
        traget_input_ids = traget_idxs['input_ids'].to(device)
        traget_attention_mask = traget_idxs['attention_mask'].to(device)
        traget_token_type_ids = traget_idxs['token_type_ids'].to(device)
        traget_pred = model(traget_input_ids, traget_attention_mask, traget_token_type_ids)

        source_input_ids = source_idxs['input_ids'].to(device)
        source_attention_mask = source_idxs['attention_mask'].to(device)
        source_token_type_ids = source_idxs['token_type_ids'].to(device)
        source_pred = model(source_input_ids, source_attention_mask, source_token_type_ids)

        similarity_list = F.cosine_similarity(traget_pred, source_pred)
        similarity_list = similarity_list.cpu().numpy()
    return similarity_list
    
def cal_coef(similarity_list, label_list):
    label_list = np.array(label_list)
    corrcoef = scipy.stats.spearmanr(label_list, similarity_list).correlation
    return corrcoef                        

def test(test_data, model):
    traget_idxs, source_idxs, label_list = test_data.get_data()
    with torch.no_grad():
        traget_input_ids = traget_idxs['input_ids'].to(device)
        traget_attention_mask = traget_idxs['attention_mask'].to(device)
        traget_token_type_ids = traget_idxs['token_type_ids'].to(device)
        traget_pred = model(traget_input_ids, traget_attention_mask, traget_token_type_ids)

        source_input_ids = source_idxs['input_ids'].to(device)
        source_attention_mask = source_idxs['attention_mask'].to(device)
        source_token_type_ids = source_idxs['token_type_ids'].to(device)
        source_pred = model(source_input_ids, source_attention_mask, source_token_type_ids)

        similarity_list = F.cosine_similarity(traget_pred, source_pred)
        similarity_list = similarity_list.cpu().numpy()
        label_list = np.array(label_list)
        corrcoef = scipy.stats.spearmanr(label_list, similarity_list).correlation
    return corrcoef


if __name__ == '__main__':
    model = model(model_path).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
#     # data
    training_data = TrainDataset(train_data, tokenizer, args.maxlen)
    train_dataloader = DataLoader(training_data,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=16)
    testing_data = TestDataset(test_data, tokenizer, args.maxlen)
#     # train
    train(train_dataloader, testing_data, model, optimizer)
    print("train finish!")

    # test
    model.load_state_dict(torch.load(args.save_path))
    model.eval()
    
    # 验证集效果
    traget_idxs, source_idxs, label_list = testing_data.get_data()
    similarity_list = cal_sim(traget_idxs, source_idxs, model)
#     print("test sim: {},".format(len(similarity_list)), similarity_list)
#     print("test label: {},".format(len(label_list)), label_list)
    corrcoef = cal_coef(similarity_list, label_list)
    print("test corrcoef: {}".format(corrcoef))
    
    # 预测集效果
    similarity_list, label_lists = [], []
    for i in tqdm(range(0, len(pred_data), 1000)):
        preding_data = TestDataset(pred_data[i:i+1000], tokenizer, args.maxlen)
        traget_idxs, source_idxs, label_list = preding_data.get_data()
        similarity_list.extend(cal_sim(traget_idxs, source_idxs, model))
        label_lists.extend(label_list)
    corrcoef = cal_coef(similarity_list, label_lists)
    print("pred corrcoef: {}".format(corrcoef))
    
    # 保存预测值
    with open(pred_file_path + ".pred.SimCSE.supervised.kg_14d_1w", 'w') as wf:
        for (label, prob) in zip(label_lists, similarity_list):
            wf.write(str(label) + '\t' + str(prob) + '\n')