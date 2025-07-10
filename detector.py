import numpy as np
import torch
from model import Autoencoder


def detect_anomalies(model, data_loader, threshold):
    model.eval()
    anomalies = []
    reconstruction_errors = []

    with torch.no_grad():
        for data in data_loader:
            # 取出测试数据集
            inputs, = data  # 从元组中解包
            outputs = model(inputs)
            mse = torch.mean((outputs - inputs) ** 2, dim=1)
            # 重构误差这里使用均方误差
            reconstruction_errors.extend(mse.numpy())
            #比较的结果是一个布尔类型的 torch.Tensor，其中每个元素对应一个样本，
            # 若该样本的重构误差大于阈值，则对应元素为 True，反之则为 False。
            anomalies.extend((mse > threshold).numpy())

    return np.array(anomalies), np.array(reconstruction_errors)


def determine_threshold(model, train_loader):
    model.eval()
    reconstruction_errors = []

    with torch.no_grad():
        for data in train_loader:
            inputs, = data
            outputs = model(inputs)
            # 计算均方误差，异常检测阶段不需要进行反向传播，因此不需要使用 torch.nn.MSELoss() 提供的自动求导功能。
            # 直接使用基本的张量操作计算 MSE 可以避免不必要的计算开销，提高检测效率。
            mse = torch.mean((outputs - inputs) ** 2, dim=1)
            reconstruction_errors.extend(mse.numpy())

    # 阈值为重构误差的均值加上3倍重构误差的标准差
    threshold = np.mean(reconstruction_errors) + np.std(reconstruction_errors) * 3
    return threshold, np.array(reconstruction_errors)