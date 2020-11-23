# coding=utf-8

from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import torchvision
import torch
import torchvision.transforms as transforms
import gzip
import numpy as np
import os
import matplotlib.pyplot as plt


def d_relu(x):
    '''
    relu激活函数的导数
    '''
    d = torch.zeros_like(x)
    d[x > 0] = 1
    return d


def d_softmax(x):
    '''
    softmax激活函数的导数
    '''
    d = torch.softmax(x, dim=1)
    return d*(1-d)


def load_mnist(path, kind='train'):
    """
    Load MNIST data from `path`
    """
    labels_path = os.path.join(path, '%s-labels-idx1-ubyte.gz'% kind)
    images_path = os.path.join(path, '%s-images-idx3-ubyte.gz'% kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)
    
    features = transforms.ToTensor()(images)  # (h, w, c) -> (c, h, w)
    labels = torch.LongTensor(labels)

    return features[0], labels


class FNN:
    def __init__(self, features, labels, params, prob_dropout, batch_size=256):
        '''
        features: 特征
        labels: 标签
        params元素: (权重矩阵, 偏置向量, 激活函数, 激活函数的导数)
        注意: 权重的形式为 (M_{l}, M_{l-1}), 偏置的形式为 (1, M_{l})
        '''
        self.features = features
        self.labels = labels
        self.params = params
        self.prob_dropout = prob_dropout  # 对隐藏层进行dropout操作
        self.train_iter = self.data_loader(batch_size=256)
    
    def data_loader(self, batch_size, is_one_hot=True):
        '''
        构建小批量数据集
        '''
        if is_one_hot:
            hot_labels = torch.zeros(self.features.shape[0], 10)
            x_indices = np.arange(self.features.shape[0]).tolist()
            y_indices = labels.byte().tolist()
            hot_labels[x_indices, y_indices] = 1
            dataset = TensorDataset(self.features, hot_labels)
        else:
            dataset = TensorDataset(self.features, self.labels)

        return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    
    def mask(self, X, p):
        '''
        X: 输入
        p: 神经元的保留概率
        若输入X除以概率p，则使X的期望保持不变, 预测时, 神经网络的权重不用转换；
        若训练时按正常的输入训练X，预测时，神经网络中对应的权重需乘p
        '''
        if p == 0:
            return torch.zeros_like(X)
        elif p == 1:
            return X
        else:
            prob = p*torch.ones_like(X)
            return X*torch.bernoulli(prob)/p
    
    def train_forward(self, X):
        '''
        训练用神经元前馈传递信息: 依照概率p随机关闭一些神经元
        '''
        y = X  # 初始化输入features
        mask_y = self.mask(y, self.prob_dropout[0])
        self.z_list, self.a_list = [], [y]  # 记录各层的净值和激活值
        for i, (weight, bias, func, _) in enumerate(self.params, start=1):
            p = self.prob_dropout[i]
            z = y@torch.transpose(weight, 0, 1) + bias.reshape(1, -1)  # (N, M_{l-1}) @ (M_{l-1}, M_{l}) + (1, M_{l}), broadcast
            mask_z = self.mask(z, p)
            if func:
                if func.__name__ == 'softmax':
                    y = func(mask_z, dim=1)
                else:
                    y = func(mask_z)
            else:
                y = mask_z
            
            self.z_list.append(mask_z)
            self.a_list.append(y)
        return y.double()
    
    def predict_forward(self, X):
        '''
        预测用神经元前馈传递信息: 使用所有神经元
        '''
        y = X  # 初始化输入features
        for i, (weight, bias, func, _) in enumerate(self.params):
            z = y@torch.transpose(weight, 0, 1) + bias.reshape(1, -1)
            if func:
                if func.__name__ == 'softmax':
                    y = func(z, dim=1)
                else:
                    y = func(z)
            else:
                y = z

        return y.double()
    
    def cross_entropy(self, y, hat_y):
        '''
        采用交叉熵损失函数
        y: one-hot形式
        hat_y: softmax之后对应概率向量，多层感知机的输出
        '''
        if len(y.shape) == 2:
            crossEnt = -torch.dot(y.reshape(-1), torch.log10(hat_y.float()).reshape(-1)) / y.shape[0]  # 展开成1维，点积
        elif len(y.shape) == 1:
            crossEnt = -torch.mean(torch.log10(hat_y[torch.arange(y.shape[0]), y.long()]))
        else:
            print("Wrong format of y!")
        return crossEnt
    
    def cal_neuron_errors(self, y):
        '''
        计算神经元的误差
        '''
        # 输出层误差
        error_L = self.a_list[-1] - y
        self.error_list = [error_L]
        for i in range(len(self.params)-1):
            weight = self.params[-i-1][0]  # 权重矩阵
            der_f = self.params[-i-1][-1]  # 导数
            error_up = self.error_list[-1]  # 上一层的误差
            z = self.z_list[-i-2]  # 当前层的净值
            error = error_up@weight * der_f(z)  #  (N, M_{l})@(M_{l}, M_{l-1}) = (N, M_{l-1})
            self.error_list.append(error)
            
        self.error_list.reverse()
    
    def cal_params_partial(self):
        '''
        计算损失函数关于权重和偏置的偏导数
        '''
        self.der_weight_list = []
        self.der_bias_list = []
        for i in range(len(self.params)):
            a_out = self.a_list[i]
            error_in = self.error_list[i]
            # 以下计算出来的是每个样本对应的der_weight构成的矩阵，归约成1维，可采用均值或求和的形式
            der_weight = torch.transpose(error_in, 0, 1)@a_out / self.a_list[0].shape[0]  # (M_{l}, N) @ (N, M{l-1})
            der_bias = torch.mean(torch.transpose(error_in, 0, 1), axis=1)  # (M_{l}, N)
            self.der_weight_list.append(der_weight)
            self.der_bias_list.append(der_bias)
        
    def backward(self, y):
        '''
        误差反向传播算法实现
        '''
        self.cal_neuron_errors(y)
        self.cal_params_partial()
    
    def accuracy(self, y, hat_y, is_one_hot=False):
        '''
        y: 标签, one-hot
        hat_y: 标签预测概率, one-hot
        is_one_hot: y是否为one-hot形式
        '''
        if is_one_hot:
            precision = torch.sum(torch.max(y, axis=1)[1] == torch.max(hat_y, axis=1)[1]).numpy() / y.shape[0]
        else:
            precision = torch.sum((y == torch.max(hat_y, axis=1)[1]).byte()).numpy() / y.shape[0]
        return precision
    
    def minibatch_sgd_trainer(self, max_epochs=10, lr=0.1, decay=0.0005):
        '''
        训练
        lr: 学习率
        decay: 权重衰减系数
        '''
        for epoch in range(max_epochs):
            for X, y in self.train_iter:
                self.train_forward(X)  # 前向传播
                self.backward(y)  # 误差反向传播
                for i in range(len(self.params)):
                    self.params[i][0] = (1 - decay)*self.params[i][0] - lr*self.der_weight_list[i]
                    self.params[i][1] = (1 - decay)*self.params[i][1] - lr*self.der_bias_list[i]
            
            hat_labels = self.predict_forward(self.features)
            loss = self.cross_entropy(self.labels, hat_labels)
            accu = self.accuracy(self.labels, hat_labels)
            print(f"第{epoch+1}个回合, 训练集交叉熵损失为:{loss:.4f}, 分类准确率{accu:.4f}")


if __name__ == "__main__":
    # 标签
    label_names = ['短袖圆领T恤', '裤子', '套衫', '连衣裙', '外套', '凉鞋', '衬衫', '运动鞋','包', '短靴']
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    # 加载数据
    features, labels = load_mnist(path="../dataset/fashion_mnist")
    test_features, test_labels = load_mnist(path="../dataset/fashion_mnist", kind="t10k")
    # 参数初始化
    num_inputs, num_outputs, num_hiddens = 784, 10, 256
    W1 = torch.tensor(np.random.normal(0, 2/num_inputs, (num_hiddens, num_inputs)), dtype=torch.float)
    b1 = torch.zeros(num_hiddens, dtype=torch.float)
    W2 = torch.tensor(np.random.normal(0, 2/num_hiddens, (num_outputs, num_hiddens)), dtype=torch.float)
    b2 = torch.zeros(num_outputs, dtype=torch.float)
    init_params = [[W1, b1, torch.relu, d_relu], [W2, b2, torch.softmax, d_softmax]]
    prob_dropout = [0.95, 0.5, 1]  # [输入层, 隐藏层, 输出层]
    # 实例化
    fnn = FNN(features, labels, init_params, prob_dropout)
    # 训练
    fnn.minibatch_sgd_trainer(max_epochs=40, lr=0.5, decay=0)
    # 预测集表现
    hat_test_labels = fnn.predict_forward(test_features)
    test_crossEn = fnn.cross_entropy(test_labels, hat_test_labels)
    test_accu = fnn.accuracy(test_labels, hat_test_labels, is_one_hot=False)
    print(f"测试集上的交叉熵为{test_crossEn:.4f}, 测试集的准确率为:{test_accu:.4f}")
    