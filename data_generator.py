import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset


def generate_normal_data(n_samples=1000, n_features=10):
    # 生成均值为0，标准差为1，（标准差越小，越集中在均值附近）的正常数据
    return np.random.normal(loc=0, scale=1, size=(n_samples, n_features))


def generate_anomalous_data(n_samples=100, n_features=10):
    return np.random.normal(loc=5, scale=3, size=(n_samples, n_features))


def prepare_data():
    # 生成数据
    normal_data = generate_normal_data(n_samples=1000, n_features=10)
    anomalous_data = generate_anomalous_data(n_samples=100, n_features=10)

    # 划分训练集和测试集
    # 从正常数据中划分出0.2的测试样本
    # 这里的正常样本不能直接用训练集剩下的，而是从原始正常数据中单独拆分（X_test_normal），确保测试数据是模型 “没见过” 的，否则评估结果会失真
    X_train, X_test_normal = train_test_split(normal_data, test_size=0.2, random_state=42)

    # 合并测试数据
    X_test = np.vstack([X_test_normal, anomalous_data])
    # 打标签，正常数据位0，异常数据位1
    y_test = np.hstack([np.zeros(len(X_test_normal)), np.ones(len(anomalous_data))])

    # 转换为PyTorch张量
    X_train_tensor = torch.FloatTensor(X_train)
    X_test_tensor = torch.FloatTensor(X_test)

    # 创建数据加载器
    train_dataset = TensorDataset(X_train_tensor)
    test_dataset = TensorDataset(X_test_tensor)

    # shuffle=True：训练集打乱顺序（增强泛化能力）
    # shuffle=False：测试集保持顺序（便于后续与标签对应）
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_loader, test_loader, X_test, y_test