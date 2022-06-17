from train_model.model import FaceCNN
import torch
import numpy as np
import random
import cv2
import os

def search_files(directory):
    directory = os.path.normpath(directory)
    datas = []
    for curdir, subdirs, files in os.walk(directory):
        for file in files:
            data_label = []
            if file.endswith('.jpg'):
                label = curdir.split(os.path.sep)[-1]
                path = os.path.join(curdir, file)
                data_label.append(path)
                data_label.append(label)
                datas.append(data_label)
                random.shuffle(datas)

    return datas

def get_result(file_path, result):
    model = FaceCNN()
    model.load_state_dict(torch.load("model_net_result.pth"))
    image = cv2.imread(file_path[0])
    # 读取单通道灰度图
    if image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 直方图均衡化
    image_hist = cv2.equalizeHist(image)
    image_normalized = image_hist.reshape(1, 64, 64) / 255.0
    image_tensor = torch.from_numpy(image_normalized)
    image_tensor = image_tensor.type('torch.FloatTensor')
    image_tensor_ = torch.unsqueeze(image_tensor, 0)

    pred = model.forward(image_tensor_)
    pred = np.argmax(pred.data.numpy(), axis=1)
    print(result[pred[0]])
    if int(pred[0]) == int(file_path[1]):
        print("Y")
        cv2.imshow("face", image)
        cv2.waitKey(100)
    else:
        print("N")
        cv2.imshow("face", image)
        cv2.waitKey(3000)



if __name__ == "__main__":
    result = ["非笑脸", "笑脸"]
    file_path = "C:\\Users\\bhj\\Desktop\\smile_train\\datasets\\val_data"
    # get_result(file_path, result)
    images_path = search_files(file_path)
    for file_p in images_path:
        get_result(file_p, result)