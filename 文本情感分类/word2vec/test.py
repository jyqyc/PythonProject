import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.utils.data as Data

dtype = torch.FloatTensor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 文本预处理
sentences = ["jack like dog", "jack like cat", "jack like animal",
             "dog cat animal", "banana apple cat dog like", "dog fish milk like",
             "dog cat animal like", "jack like apple", "apple like", "jack like banana",
             "apple banana jack movie book music like", "cat dog hate", "cat dog like"]

word_sequence = " ".join(sentences).split()  # ['jack', 'like', 'dog', 'jack', 'like', 'cat', 'animal',...]
vocab = list(set(word_sequence))  # build words vocabulary，去重
word2idx = {w: i for i, w in enumerate(vocab)}  # {'apple': 0, 'fish': 1,..., }，注意，不固定！！！

# 模型的相关参数
batch_size = 8
embedding_size = 2  # 词向量的维度是2
C = 2  # window size
voc_size = len(vocab)

# 数据预处理
skip_grams = []
print(word2idx)
for idx in range(C, len(word_sequence) - C):
    center = word2idx[word_sequence[idx]]  # 中心词

    context_idx = list(range(idx - C, idx)) + list(range(idx + 1, idx + C + 1))  # 中心词左边的2个词+中心词右边的两个词
    context = [word2idx[word_sequence[i]] for i in context_idx]
    for w in context:
        skip_grams.append([center, w])  # 中心词和每个周围词组成一个训练样本


def make_data(skip_grams):
    input_data = []
    output_data = []
    for i in range(len(skip_grams)):
        # input_data转换为one-hot形式，output_data合成一个list
        input_data.append(np.eye(voc_size)[skip_grams[i][0]])
        output_data.append(skip_grams[i][1])
    return input_data, output_data


print(skip_grams)
input_data, output_data = make_data(skip_grams)
print(input_data)
print(output_data)
input_data, output_data = torch.Tensor(np.array(input_data)), torch.LongTensor(np.array(output_data))
dataset = Data.TensorDataset(input_data, output_data)
loader = Data.DataLoader(dataset, batch_size, True)
"""
skip_grams: [[10, 2],[9, 8], [11, 5], ..., [11, 7], [11, 10], [11, 0]]
input_data: [array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.]),...]
output_data: [2, 0, 2, 0, 0, 10, 0, 11, 10, 2, 11, 2, 2, 0, 2, 0, 0, 11, 0, 8, 11, 2, 8, 10, 2, 0, 10,...]
"""


# 构建模型
class Word2Vec(nn.Module):
    def __init__(self):
        super(Word2Vec, self).__init__()
        self.W = nn.Parameter(torch.randn(voc_size, embedding_size).type((dtype)))
        self.V = nn.Parameter(torch.randn(embedding_size, voc_size).type((dtype)))

    def forward(self, X):
        # X : [batch_size, voc_size] one-hot
        # torch.mm only for 2 dim matrix, but torch.matmul can use to any dim
        hidden_layer = torch.matmul(X, self.W)  # hidden_layer : [batch_size, embedding_size]
        output_layer = torch.matmul(hidden_layer, self.V)  # output_layer : [batch_size, voc_size]
        return output_layer


model = Word2Vec().to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 训练
for epoch in range(2000):
    for i, (batch_x, batch_y) in enumerate(loader):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        pred = model(batch_x)
        loss = criterion(pred, batch_y)
        if (epoch + 1) % 1000 == 0:
            print(epoch + 1, i, loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# 将每个词在平面直角坐标系中标记出来，看看各个词之间的距离
for i, label in enumerate(vocab):
    W, WT = model.parameters()
    # W是词向量矩阵
    x, y = float(W[i][0]), float(W[i][1])
    plt.scatter(x, y)
    plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
plt.show()

# https://wmathor.com/index.php/archives/1443/