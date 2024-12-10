# 划分数据集为测试集和训练集
# 随机抽样
import csv
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def read_data1(datapath):
    data = pd.read_csv(datapath, encoding="utf-8")
    x_list = data['Text']
    y_list = data['Translation']

    return np.array(x_list), np.array(y_list)


def split_data(data_list, y_list, ratio=0.20):  # 80%训练集，20%测试集
    '''
    按照指定的比例，划分样本数据集
    ratio: 测试数据的比率
    '''
    X_train, X_test, y_train, y_test = train_test_split(data_list, y_list, test_size=ratio, random_state=50)

    """训练集"""
    with open('./trainDataset.csv', 'w', encoding="utf_8_sig",
              newline="",
              errors="ignore") as csvfile:  # 不加newline=""的话会空一行出来
        fieldnames = ['Text', 'Translation']
        write = csv.DictWriter(csvfile, fieldnames=fieldnames)
        write.writeheader()  # 写表头
        for i in range(len(X_train)):
            write.writerow({'Text': X_train[i], 'Translation': y_train[i]})

    """测试集"""
    # 测试csv
    with open('./testDataset.csv', 'w', encoding="utf_8_sig",
              newline="",
              errors="ignore") as csvfile:  # 不加newline=""的话会空一行出来
        fieldnames = ['Text', 'Translation']
        write = csv.DictWriter(csvfile, fieldnames=fieldnames)
        write.writeheader()  # 写表头
        for i in range(len(X_test)):
            write.writerow({'Text': X_train[i], 'Translation': y_train[i]})
    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    datapath = './shieldTunnelDesignStandard-WithPrologTranslation4.csv'
    """获取大文件的数据"""
    x_list, y_list = read_data1(datapath)
    """划分为训练集和测试集及label文件"""
    split_data(x_list, y_list, ratio=0.10)