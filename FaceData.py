import torch
import torch.utils.data as data
import numpy as np
import pandas as pd
import cv2

class FaceDataset(data.Dataset):
    # 初始化
    def __init__(self, root, train_csv):
        super(FaceDataset, self).__init__()
        self.root = root
        image_data = pd.read_csv(train_csv, header=None, usecols=[0])
        label_data = pd.read_csv(train_csv, header=None, usecols=[1])
        self.path = np.array(image_data)[:, 0]
        self.label = np.array(label_data)[:, 0]

    # 读取某幅图片，item为索引号
    def __getitem__(self, item):
        # 图像数据用于训练，需为tensor类型，label用numpy或list均可
        face = cv2.imread(self.root + "\\" + str(self.label[item]) + "\\" + self.path[item])
        """
        像素值标准化
        读出的数据是64X64的，而后续卷积神经网络中nn.Conv2d() API所接受的数据格式是(batch_size, channel, width, higth)，
        本次图片通道为3，因此我们要将64X64 reshape为3X64X64。
        """
        # 读取单通道灰度图
        if face.shape[2] == 3:
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        # 直方图均衡化
        face_hist = cv2.equalizeHist(face)
        face_normalized = face_hist.reshape(1, 64, 64) / 255.0
        face_tensor = torch.from_numpy(face_normalized)
        face_tensor = face_tensor.type('torch.FloatTensor')
        label = self.label[item]
        return face_tensor, label

    # 获取数据集样本个数
    def __len__(self):
        return self.path.shape[0]

if __name__ == "__main__":
    data_train_path = "C:\\Users\\bhj\\Desktop\\smile_train\\datasets\\train_data"
    data_val_path = "C:\\Users\\bhj\\Desktop\\smile_train\\datasets\\val_data"
    train_csv = "C:\\Users\\bhj\\Desktop\\smile_train\\train_data.csv"
    val_csv = "C:\\Users\\bhj\\Desktop\\smile_train\\val_data.csv"

    datasss = FaceDataset(data_path, val_csv)
    train_loader = data.DataLoader(datasss, 1)
    for i in range(10000):
        for images, labels in train_loader:
            # print(images)
            print(labels)
        