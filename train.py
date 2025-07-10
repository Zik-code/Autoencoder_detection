import torch
import torch.optim as optim
from model import Autoencoder


def train_autoencoder(model, train_loader, num_epochs=50):
    # 创建损失实例
    criterion = torch.nn.MSELoss()
    # 创建优化实例
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for data in train_loader:
            inputs, = data  # 从元组中解包
            # 每个批次训练前，梯度清零
            optimizer.zero_grad()
            # 计算损失
            loss = criterion(model(inputs), inputs)
            loss.backward()
            # 优化器
            optimizer.step()
            running_loss += loss.item()

            # 假设数据总共100个（train_loader），训练10轮，每一轮中都训练这100个数据，每次取出10个(data)数据进行一批次训练
            # running_loss就是训练一轮的损失

        # 训练完轮次后打印信息
        if (epoch + 1) % 10 == 0:
            # 总损失除一轮的损失批次数
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.6f}')

    return model

