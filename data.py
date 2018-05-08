import os
import numpy as np
import torch
import torch.utils.data as data
import cv2
from PIL import Image
from tqdm import tqdm

# 训练集与测试集所在目录
sep = os.sep
trainpath = 'train'
testpath = 'test1'

# 训练集是分子目录的，且子目录是类标号，这里得到所有子目录（类别）及其数目
categories = os.listdir(trainpath)
category_number = len(categories)

# 后续用到卷积网络，这里设置图片的大小，之后需要统一将图片都reshape到这个尺寸
img_size = (128, 128)


def load_by_category(category_index):
    '''
    :param category_index:
    :return: samples 一个
    '''
    # 目录拼接
    path = trainpath + sep + categories[category_index]
    # 获取所有该目录（类别）的样本（图片文件）
    files = os.listdir(path)
    # 结果集
    samples = []
    # 遍历所有该类样本
    for file in files:
        # 目录拼接
        file = path + sep + file
        # 读取图片到np数组
        img = np.asarray(Image.open(file))
        # 将图片reshape到128*128
        img = cv2.resize(img, img_size)
        # 加入结果集
        samples.append(img)
    samples = np.array(samples)
    # 类标号：每一幅图片的结果都是一个长度为category_number的结果向量
    labels = np.zeros([len(samples), category_number], dtype=np.uint8)
    # 当前labels的类标号设置为1，通过位置确定类
    # 可以想象最后将所有的category的labels拼接起来应该是一个阶梯矩阵
    labels[:, category_index] = 1
    return samples, labels


def trans_data():
    # 创建一个空的n*128*128的数据集合,第一维表示样本索引，第二三维128*128表示数据
    # 这里*img_size是列表的元素解包，这样[-1,*img_size]等于[-1,128,128]，而不是[-1,(128,128)]
    all_samples = np.zeros([0, *img_size], dtype=np.uint8)
    # 创建一个空的n*100的结果（类标号）集合，第一维表示样本index，100表示该样本属于这100个类别的概率（训练集上是1和0）
    labels = np.reshape(np.array([], dtype=np.uint8), [0, category_number])
    # 遍历所有训练集子目录（类别），这里tqdm是一个可视化进度条的库
    for index in tqdm(range(category_number)):
        # 获取某一个类别的数据和标注
        data, label = load_by_category(index)
        # 将结果合并到最终结果集中
        all_samples = np.append(all_samples, data, axis=0)
        labels = np.append(labels, label, axis=0)
    # 保存数据，all_samples - 100*
    np.save('data.npy', all_samples)
    np.save('label.npy', labels)


class TrainSet(data.Dataset):
    def __init__(self, eval=False):
        datas = np.load('data.npy')
        labels = np.load('label.npy')
        index = np.arange(0, len(datas), 1, dtype=np.int)
        np.random.seed(123)
        # index是一个有序列表，长度为samples的数量，shuffle是打乱这个索引列表
        np.random.shuffle(index)
        if eval:
            index = index[:int(len(datas) * 0.1)]
        else:
            index = index[int(len(datas) * 0.1):]
        self.data = datas[index]
        self.label = labels[index]
        np.random.seed()

    def __getitem__(self, index):
        return torch.from_numpy(self.data[index]), torch.from_numpy(self.label[index])

    def __len__(self):
        return len(self.data)


def load_test_data():
    files = os.listdir(testpath)
    test_data_set = []
    for file in tqdm(files):
        file = testpath + sep + file
        img = np.asarray(Image.open(file))
        img = cv2.resize(img, img_size)
        test_data_set.append(img)
    test_data_set = np.array(test_data_set)
    return test_data_set


if __name__ == '__main__':
    # trans_data()
    train_data = np.load('data.npy')
    train_labels = np.load('label.npy')
    print(train_data.shape, train_labels.shape)
    test_data = load_test_data()
    print(test_data.shape)
