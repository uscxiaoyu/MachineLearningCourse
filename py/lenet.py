from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import torch
import torchvision.transforms as transforms
import numpy as np
import gzip
import os
import time


class LeNet(nn.Module):
    """
    输入数据X的格式为: (N, C, H, W), N表示样本量, C为通道量, H为高, W为宽
    """

    def __init__(self):
        super(LeNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, 5),  # in_channels, out_channels, kernel_size 6*(28-5+1)*(28-5+1)
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2),  # kernel_size, stride 6*(24/2)*(24/2)
            nn.Conv2d(6, 16, 5),  # 16*(12-5+1)*(12-5+1)
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2),  # 16*(8/2)*(8/2)
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * 4 * 4, 120), nn.Sigmoid(), nn.Linear(120, 84), nn.Sigmoid(), nn.Linear(84, 10)
        )

    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        return output


def load_mnist(path, kind="train"):
    """
    Load MNIST data from `path`
    """
    labels_path = os.path.join(path, "%s-labels-idx1-ubyte.gz" % kind)
    images_path = os.path.join(path, "%s-images-idx3-ubyte.gz" % kind)

    with gzip.open(labels_path, "rb") as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

    with gzip.open(images_path, "rb") as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)

    features = transforms.ToTensor()(images)  # (h, w, c) -> (c, h, w)
    labels = torch.LongTensor(labels)

    return features[0], labels


def data_loader(features, labels, batch_size, is_one_hot=True):
    """
    构建小批量数据集
    """
    features = features.view(features.shape[0], 1, 28, 28)  # 设置为 N*C*H*W
    if is_one_hot:
        hot_labels = torch.zeros(features.shape[0], 10)
        x_indices = np.arange(features.shape[0]).tolist()
        y_indices = labels.byte().tolist()
        hot_labels[x_indices, y_indices] = 1
        dataset = TensorDataset(features, hot_labels)
    else:
        dataset = TensorDataset(features, labels)

    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)


def evaluate_accuracy(data_iter, net, device=None):
    if device is None and isinstance(net, torch.nn.Module):
        # 如果没指定device就使用net的device
        device = list(net.parameters())[0].device
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(net, torch.nn.Module):
                net.eval()  # 评估模式, 这会关闭dropout
                acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
                net.train()  # 改回训练模式
            else:  # 自定义的模型, 不考虑GPU
                if "is_training" in net.__code__.co_varnames:  # 如果有is_training这个参数
                    # 将is_training设置成False
                    acc_sum += (net(X, is_training=False).argmax(dim=1) == y).float().sum().item()
                else:
                    acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
            n += y.shape[0]
    return acc_sum / n


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path = "C:/Users/Yu_HomeStudio/GitProjects/MachineLearningCourse/dataset/mnist"
    features, labels = load_mnist(path=path)
    test_features, test_labels = load_mnist(path=path, kind="t10k")
    batch_size = 256
    train_iter = data_loader(features, labels, batch_size, is_one_hot=False)
    test_iter = data_loader(test_features, test_labels, batch_size, is_one_hot=False)

    num_epochs = 40
    net = LeNet()
    net = net.to(device)
    print("training on ", device)
    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.perf_counter()
        for X, y in train_iter:
            X = X.to(device)  # 将张量复制到设备device
            y = y.to(device)  # 将张量复制到设备device
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        test_acc = evaluate_accuracy(test_iter, net)
        print(
            "epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec"
            % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.perf_counter() - start)
        )

