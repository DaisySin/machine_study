## 导入工具包

import torch
import logging
import shutil
import numpy as np
import time
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
logger = logging.getLogger('main')

batch_size = 64
max_length = 40   # 前提或假设最大句子长度
lstm_hid_dim = 128 # lstm内部神经cell个数
word_dim=100 #词向量大小
n_class=3 #假设label类别数量
eval_best_f1=0 ##用于标记训练过程正确率最优模型
eval_best_accuracy=0##用于标记训练过程f1最优模型
train_best_accuracy_model=''##记录正确率最优的模型存储路径
train_best_f1_model='' ##记录f1最优的模型的存储路径
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')##有GPU用GPU，无GPU用CPU
train_epoches = 2 #训练轮数
dropout =0.6 # dropout 保留率

########### 1.数据准备 #############
#1.1 原始数据预处理
import json
import re
import re

def tokens(data_path,file):
    with open(data_path) as fin:
        with open(file,'w') as ww:
            for lidx, line in enumerate(fin):
                sample = json.loads(line.strip())
                one={}
                if sample['gold_label']=='-':
                    continue
                one['label'] = sample['gold_label']
                sentence1 = sample['sentence1']
                seg_sen1 = re.split(r'[;,.\s]\s*', sentence1)    # 正则匹配：1、分号、逗号、空格中任意一个；2、匹配任意个空格
                sentence2 = sample['sentence2']
                seg_sen2 = re.split(r'[;,\s.]\s*', sentence2)
                one['arg1'] = seg_sen1
                one['arg2'] = seg_sen2
                ww.write(json.dumps(one)+'\n')
            print("end")
print("begin....")
tokens("datasets/raw_data/snli_1.0/snli_1.0_train.jsonl", "temp/snli_train.json")
tokens("datasets/raw_data/snli_1.0/snli_1.0_dev.jsonl", "temp/snli_dev.json")
tokens("datasets/raw_data/snli_1.0/snli_1.0_test.jsonl", "temp/snli_test.json")

#1.1.1 数据加载
def dataprocess(path,datatype=''):
    example=[]
    with open(path,'r',encoding='utf-8') as fin:
         for lidx, line in enumerate(fin):
            sample = json.loads(line.strip())
            example.append(sample)
    return example


train_example=dataprocess("./temp/snli_train.json")
dev_example=dataprocess("./temp/snli_dev.json")
test_example=dataprocess("./temp/snli_test.json")
print("训练集数量：%d"%(len(train_example)))
print("验证集数量：%d"%(len(dev_example)))
print("测试集数量：%d"%(len(test_example)))

# 1.2 构建词典加载词向量
import numpy as np
import torch
def load_embedding(embedding_file):
    """
    加载词向量，返回词典和词向量矩阵
    :param embedding_file: 词向量文件
    :return: tuple, (词典, 词向量矩阵)
    """
    with open(embedding_file, encoding='utf-8') as f:
        lines = f.readlines()
        embedding_tuple = [tuple(line.strip().split(' ', 1)) for line in lines]
        embedding_tuple = [(t[0].strip().lower(), list(map(float, t[1].split()))) for t in embedding_tuple]
    embedding_matrix = []
    embedding_dim = len(embedding_tuple[0][1])
    embedding_matrix.append([0] * embedding_dim)  # 词向量全为0，表示未登录词
    embedding_matrix.append([0] * embedding_dim)  # 词向量全为0，表示间隔词
    word_dict = dict()
    word_dict['_UNK_'] = 0  # _UNK_表示未登录词
    word_dict['_SEP_'] = 1  # _SEP_表示间隔词
    word_id = 2
    for word, embedding in embedding_tuple:
        if word_dict.get(word) is None:
            word_dict[word] = word_id
            word_id += 1
            embedding_matrix.append(embedding)
    return word_dict, np.asarray(embedding_matrix, dtype=np.float32)
vocab_to_int, embedding_matrix = load_embedding("./datasets/glove.6B.100d.txt")  # 英文词向量
embed = torch.from_numpy(embedding_matrix).float()
vocab_to_int
print("dictionary length: %d" % (len(vocab_to_int)))

#1.3 模型输入预处理

#1.3.1 word2id 转换
def wordlists2idlists(tokens,word_to_id):
    """
        句子列表转id列表的列表
        :param word_list_s: 词列表的列表
        :param word_to_id: 词典
        :return: list of ints. id形式的句子列表
        """
    ids_list = [word_to_id.get(word, 0) for word in tokens]
    return ids_list

#1.3.2 输入特征提取
labels = {'entailment': 0, 'contradiction': 1, 'neutral': 2}


def create_seq_pair(tokens_a, tokens_b):
    """
    当句子长度大于给定的最大句长时，对句子进行剪切

    """
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length - 3:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def convert_example_to_feature(example, datatype=''):
    """ 按照input_ids  label  segment_ids顺序生成输入特征

    """
    data = []
    for s in example:
        create_seq_pair(s['arg1'], s['arg2'])
        tokens = []
        for q in s['arg1']:
            tokens.append(q)
        tokens.append('_SEP_')

        for q in s['arg2']:
            tokens.append(q)

        inputs_ids = wordlists2idlists(tokens, vocab_to_int)
        while len(inputs_ids) < max_length:  ###句长不足设定长度时补0
            inputs_ids.append(0)
            x = {
                'inputs_ids': inputs_ids,
                'label': labels[s['label']]
            }
        data.append(x)
    print(len(data))
    return data

train_example_ids = convert_example_to_feature(train_example[:5000]) # 方便训练，取5000
test_example_ids = convert_example_to_feature(test_example[:800])
dev_example_ids = convert_example_to_feature(dev_example[:500])
print(train_example_ids[0])
print(test_example_ids[0])

# 2.模型构建
#2.3 分类器模块
class BasicModule(torch.nn.Module):
    def __init__(self):
        super(BasicModule, self).__init__()
        self.model_name = str(type(self))

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def save(self, path=None):
        if path is None:
            raise ValueError('Please specify the saving road!!!')
        torch.save(self.state_dict(), path)
        return path
class SelfAttention(BasicModule):
    def __init__(self, batch_size, lstm_hid_dim,n_classes,embeddings):
        super(SelfAttention, self).__init__()
        self.result = {}
        self.n_classes = n_classes
        self.embeddings = torch.nn.Embedding(
            num_embeddings=len(embeddings),
            embedding_dim=word_dim,
            padding_idx=0,
            _weight=embeddings)
        self.lstm = torch.nn.LSTM(input_size=word_dim, hidden_size=lstm_hid_dim,
                           num_layers=1, bidirectional=True,
                           dropout=dropout, batch_first=False)
        self.output_layer = torch.nn.Linear(lstm_hid_dim * 2, n_classes)
        self.embedding_dropout = torch.nn.Dropout(p=0.3)
        self.batch_size = batch_size
        self.lstm_hid_dim = lstm_hid_dim
        self.w = torch.nn.Parameter(torch.Tensor(
            lstm_hid_dim * 2, lstm_hid_dim * 2))
        self.u = torch.nn.Parameter(torch.Tensor(lstm_hid_dim * 2, 1))
        self.embeddings.weight.requires_grad = False
        self.dropout = torch.nn.Dropout(0.5)
        torch.nn.init.uniform_(self.w, -0.1, 0.1)  # 初始化w,u
        torch.nn.init.uniform_(self.u, -0.1, 0.1)

    def sef_attention(self, embeddings):
        x = embeddings.permute(1, 0, 2)
        outputs, (h, c) = self.lstm(x)
        outputs = outputs.permute(1, 0, 2)
        u = torch.tanh(torch.matmul(outputs, self.w))
        att = torch.matmul(u, self.u)
        att_score = F.softmax(att, dim=1)
        scored_x = outputs * att_score
        feat = torch.sum(scored_x, dim=1)
        return feat

    def fc_out(self, embeddings):
        embeddings = embeddings.permute(1, 0, 2)
        size = embeddings.size(1)
        h_0 = torch.randn(2, size, lstm_hid_dim).to(device)
        c_0 = torch.randn(2, size, lstm_hid_dim).to(device)
        outputs, (h_n, c_n) = self.lstm(embeddings, (h_0, c_0))
        x = h_n
        x = x.permute(1, 0, 2)  # [batch_size, num_layers*num_directions, hidden_size]
        x = x.contiguous().view(size, 2 * lstm_hid_dim)
        return x

    def forward(self, data, type=' '):
        input_ids, labels = data
        self.labels_data = labels
        embeddings = self.embeddings(input_ids)
        embeddings = self.embedding_dropout(embeddings)
        ## 实验设置
        # 这里通过对比 使用 self-attention 的区别
        feat = self.sef_attention(embeddings)  ##选择含self_attention的模型
        # feat=self.fc_out(embeddings)    ##选择不包含self_attention的模型
        output = self.output_layer(feat)
        if type == 'predict':  # 只返回结果即可
            return torch.nn.functional.softmax(output, dim=-1)
        loss = self.calc_loss(output, labels)  # 计算loss，f1，accuracy
        return loss, torch.nn.functional.softmax(output, dim=-1)

    def calc_loss(self, inputs, targets):
        # targets = targets.view(-1, 1)
        # print(inputs.shape)
        # print(targets.shape)
        loss_cross_entropy = torch.nn.CrossEntropyLoss(reduction='mean')
        loss_cross_entropy = loss_cross_entropy(inputs, targets)
        self.result['loss'] = loss_cross_entropy.detach().item()
        self.result['accuracy'] = self.calc_accuracy(inputs.detach().cpu().numpy(), targets.detach().cpu().numpy())
        self.result['f1'] = self.calc_f1(inputs.detach().cpu().numpy(), targets.detach().cpu().numpy())
        return loss_cross_entropy

    def calc_accuracy(self, inputs, targets):
        # inputs = (n, m)
        # target = (n, 1)
        outputs = np.argmax(inputs, axis=1)
        return np.mean(outputs == targets)

    def calc_f1(self, inputs, targets):
        # inputs = (n, m)
        # target = (n, 1)
        outputs = np.argmax(inputs, axis=1)
        return f1_score(targets, outputs, average='macro')

    def get_result(self):
        return self.result

    def get_labels_data(self):
        return self.labels_data

model = SelfAttention(batch_size=batch_size, lstm_hid_dim=lstm_hid_dim, n_classes=n_class, embeddings=embed)

# 3.1 数据集批处理
def get_train_loader(example):
    all_input_ids=torch.tensor([s['inputs_ids'] for s in example],dtype=torch.long)
    all_label_ids=torch.tensor([int(s['label']) for s in example],dtype=torch.long)
    data = TensorDataset(all_input_ids,all_label_ids)
    sampler=RandomSampler(data)
    data_loader = DataLoader(data,sampler=sampler,batch_size=batch_size)
    return data_loader

def get_test_loader(example):
    all_input_ids = torch.tensor([s['inputs_ids'] for s in example], dtype=torch.long)
    all_label_ids = torch.tensor([int(s['label']) for s in example], dtype=torch.long)
    data = TensorDataset(all_input_ids,all_label_ids)
    data_loader = DataLoader(data, batch_size=batch_size)
    return data_loader
def get_dev_loader(example):
    all_input_ids = torch.tensor([s['inputs_ids'] for s in example], dtype=torch.long)
    all_label_ids = torch.tensor([int(s['label']) for s in example], dtype=torch.long)
    data = TensorDataset(all_input_ids,all_label_ids)
    data_loader = DataLoader(data, batch_size=batch_size)
    return data_loader
train_loader=get_train_loader(train_example_ids)
test_loader=get_test_loader(test_example_ids)
dev_loader=get_dev_loader(dev_example_ids)

#3.2 定义优化器
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.99))


#3.3 模型训练
def model_eval(model, data_loader, data_type):
    """模型验证：

    """
    result_sum = {}
    nm_batch = 0
    labels_pred = np.array([])
    labels_true = np.array([])
    for step, batch in enumerate(data_loader):  # 按batch提取验证集数据
        batch = tuple(t.to(device) for t in batch)
        model.eval()  # 不启用 BatchNormalization 和 Dropout
        with torch.no_grad():
            _, pred = model(batch)  # 模型验证
        pred = np.argmax(pred.detach().cpu().numpy(), axis=1)  # 根据最大概率对应的下标得到对应类别
        labels_pred = np.append(labels_pred, pred)  # 得到预测结果
        true = model.get_labels_data().detach().cpu().numpy()  # 得到正确标签
        labels_true = np.append(labels_true, true)
        result_temp = model.get_result()  # 得到该批次的loss值，f1，正确率
        result_sum['loss'] = result_sum.get('loss', 0) + result_temp['loss']  # 计算各批次loss值之和
        nm_batch += 1
    """计算结果的正确率、f1值、loss值"""
    result_sum["accuracy"] = accuracy_score(labels_true, labels_pred)  # 计算正确率
    result_sum["f1"] = f1_score(labels_true, labels_pred, average='macro')  # 计算宏平均f1值
    result_sum["loss"] = result_sum["loss"] / nm_batch  # 计算平均loss值
    """将正确率、f1值、loss值存入到结果文件中"""
    with open('./temp/' + data_type + '_result.txt', 'w', encoding='utf-8') as writer:
        print("***** Eval results in " + data_type + "*****")
        for key in sorted(result_sum.keys()):
            print("%s = %s" % (key, str(result_sum[key])))
            writer.write("%s = %s\n" % (key, str(result_sum[key])))
        writer.write('\n')
    return result_sum

def save_best_model(model,v,use_accuracy=False):
    """保存最优模型"""
    if not use_accuracy:#用f1值作为评估标准
        global eval_best_f1
        if eval_best_f1<v:#保存最佳f1值模型
            eval_best_f1=v
            state = {'net': model.state_dict()}
            save_path ='./temp/_state_dict_dev' +'_f1_' + str(v) + '.model'
            print("Save.......")
            torch.save(state, save_path)#存储模型
            global train_best_f1_model
            train_best_f1_model = save_path
    if use_accuracy:#用正确率作为评估标准
        global eval_best_accuracy
        if eval_best_accuracy < v:#保存最佳正确率模型
            eval_best_accuracy = v
            state = {'net': model.state_dict()}
            save_path ='./temp/_state_dict_'+'_ac_' + str(v) + '.model'
            print("Save.......")
            torch.save(state, save_path)#存储模型
            global train_best_accuracy_model
            train_best_accuracy_model = save_path
def train(model):
    model_it_self = model.module if hasattr(model, 'module') else model
    global_step=0##统计总的batch数量
    for epoch in range(0,train_epoches):#训练train_epoches次
        for step,batch in enumerate(train_loader):#按batch提取训练数据
            global_step+=1
            model.train()#启用 BatchNormalization 和 Dropout
            batch = tuple(t.to(device) for t in batch)
            loss,output =model(batch)#模型训练
            loss.backward()#loss的反向传播
            optimizer.step()# Gardient Descent
            model.zero_grad()#把模型的参数梯度设成0
        print("epoch:%d***************"%(epoch))
        eval_result=model_eval(model_it_self,dev_loader,'dev')#用测试集验证模型效果
        save_best_model(model_it_self,eval_result['accuracy'],use_accuracy=True)##以正确率为标准保存最优模型
        save_best_model(model_it_self,eval_result['f1'])##以loss值为标准保存最优模型
    shutil.copy(train_best_accuracy_model, './temp/best_ac_model.bin')  # 更改正确率最佳模型存储路径
    shutil.copy(train_best_f1_model, './temp/best_f1_model.bin')  # 更改f1值最佳模型存储路径

model = model.to(device)
train(model)

# 4.模型测试与评估
def eval_test(model):
    best_model_path=["./temp/best_ac_model.bin", './temp/best_f1_model.bin']
    for best_model in best_model_path:
        checkpoint=torch.load(best_model)
        model.load_state_dict(checkpoint['net'], strict=False)
        model = model.to(device)
        print("\n********" + best_model + "********")
        model_eval(model, test_loader, data_type='test')
eval_test(model)

