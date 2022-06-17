import numpy as np
import pandas as pd  
import cv2 
import os 
import random


def search_files(directory):
    directory = os.path.normpath(directory)
    datas = []
    for curdir, subdirs, files in os.walk(directory):
        for file in files:
            data_label = []
            if file.endswith('.jpg'):
                label = curdir.split(os.path.sep)[-1]
                data_label.append(file)
                data_label.append(label)
                datas.append(data_label)
                random.shuffle(datas)
                # path = os.path.join(curd, file)
                # objects[label].append(path)
    train_files = []
    train_labels = []
    for data in datas:
        train_files.append(data[0])
        train_labels.append(data[1])

    return train_files, train_labels

def list_to_csv(train_files, train_labels, train_path):
    
    train_path_s = pd.Series(train_files)
    train_label_s = pd.Series(train_labels)
    train_df = pd.DataFrame()
    train_df['path'] = train_path_s
    train_df['label'] = train_label_s
    train_df.to_csv(train_path, index=False, header=False)


if __name__ == "__main__":
    train_files, train_labels = search_files("C:\\Users\\bhj\\Desktop\\smile_train\\datasets\\train_data")
    val_files, val_labels = search_files("C:\\Users\\bhj\\Desktop\\smile_train\\datasets\\val_data")
    train_path = "C:\\Users\\bhj\\Desktop\\smile_train\\train_data.csv"
    val_path = "C:\\Users\\bhj\\Desktop\\smile_train\\val_data.csv"

    list_to_csv(train_files, train_labels, train_path)
    list_to_csv(val_files, val_labels, val_path)



