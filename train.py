from train_model.model import FaceCNN
from data_set.FaceData import FaceDataset
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import cv2

# 验证模型在验证集上的正确率
def validate(model, dataset, batch_size):
    val_loader = data.DataLoader(dataset, batch_size)
    result, num = 0.0, 0
    for images, labels in val_loader:
        pred = model.forward(images)
        pred = np.argmax(pred.data.numpy(), axis=1)
        labels = labels.data.numpy()
        result += np.sum((pred == labels))
        num += len(images)
    acc = result / num
    return acc

def train(train_dataset, val_dataset, batch_size, epochs, learning_rate, wt_decay):
    # 载入数据并分割batch
    train_loader = data.DataLoader(train_dataset, batch_size)
    # 构建模型
    model = FaceCNN()
    # 损失函数
    loss_function = nn.CrossEntropyLoss()
    # 优化器
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=wt_decay)
    # 学习率衰减
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
    # 逐轮训练
    for epoch in range(epochs):
        # 记录损失值
        loss_rate = 0
        # scheduler.step() # 学习率衰减
        model.train()  # 模型训练
        for images, labels in train_loader:
            # 梯度清零
            optimizer.zero_grad()
            # 前向传播
            output = model.forward(images)
            # 误差计算
            loss_rate = loss_function(output, labels)
            # 误差的反向传播
            loss_rate.backward()
            # 更新参数
            optimizer.step()

        # 打印每轮的损失
        print('After {} epochs , the loss_rate is : '.format(epoch + 1), loss_rate.item())
        model.eval()  # 模型评估
        acc_train = validate(model, train_dataset, batch_size)
        acc_val = validate(model, val_dataset, batch_size)
        print('After {} epochs , the acc_train is : '.format(epoch + 1), acc_train)
        print('After {} epochs , the acc_val is : '.format(epoch + 1), acc_val)
        if epoch % 5 == 0:
            # model.eval()  # 模型评估
            # acc_train = validate(model, train_dataset, batch_size)
            # acc_val = validate(model, val_dataset, batch_size)
            # print('After {} epochs , the acc_train is : '.format(epoch + 1), acc_train)
            # print('After {} epochs , the acc_val is : '.format(epoch + 1), acc_val)
            torch.save(model.state_dict(), 'C:\\Users\\bhj\\Desktop\\smile_train\\moudles\\model_net%d.pth'%epoch)
    return model

def main():
    # 数据集实例化(创建数据集)
    data_train_path = "C:\\Users\\bhj\\Desktop\\smile_train\\datasets\\train_data"
    data_val_path = "C:\\Users\\bhj\\Desktop\\smile_train\\datasets\\val_data"
    train_csv = "C:\\Users\\bhj\\Desktop\\smile_train\\train_data.csv"
    val_csv = "C:\\Users\\bhj\\Desktop\\smile_train\\val_data.csv"
    train_dataset = FaceDataset(data_train_path, train_csv)
    val_dataset = FaceDataset(data_val_path, val_csv)
    # 超参数可自行指定
    model = train(train_dataset, val_dataset, batch_size=32, epochs=100, learning_rate=0.01, wt_decay=0)
    # 保存模型
    torch.save(model.state_dict(), 'model_net_result.pth')


if __name__ == '__main__':
    main()