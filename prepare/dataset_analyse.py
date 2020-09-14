# -*- coding: utf-8 -*-

from util.file_util import *
import util.function as myf
from queue import LifoQueue, Queue
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt


def is_wanted_folder(fileFolder):
    if isinstance(fileFolder, FileFolder):
        if len(fileFolder.child_files) > 0 and len(fileFolder.child_folders) == 0:
            flag = True
            for file in fileFolder.child_files:
                if not myf.str_contains(file, ".png"):
                    flag = False
                    break
            return flag
    return False



if __name__ == "__main__":
    """
    数据集分析。分析各文件夹下有多少张图像，各图像的分辨率分布情况。
    """
    dataset_dir = "D:/reid_datasets/prid_2011/multi_shot"

    # 获得只包含图像文件的直接文件夹
    folder_queue = LifoQueue()
    folder_list = []
    if os.path.isdir(dataset_dir):
        root = FileFolder(dataset_dir)
        folder_queue.put(root)
        while not folder_queue.empty():
            cur = folder_queue.get()
            if is_wanted_folder(cur):
                folder_list.append(cur.path.replace("\\", "/"))
            size = len(cur.child_folders)

            for index in range(size):
                child_folder_path = cur.child_folders[size-1-index]
                child_folder = FileFolder(child_folder_path, father=cur, FILE_FLAG=FLAG_FOLDER)
                folder_queue.put(child_folder)

            print("仍需要处理{}个文件夹".format(folder_queue.qsize()))


    # for folder in folder_list:
    #     print(folder)

    # 分析分辨率数据和各文件夹下图像分布个数
    dict_res = {}
    dict_num = {}
    image_count = 0
    for index, folder in enumerate(folder_list):
        files = os.listdir(folder)
        num_key = str(len(files))
        if (num_key in dict_num):
            dict_num[num_key] += 1
        else:
            dict_num[num_key] = 1

        for file in files:
            if myf.str_contains(file, ".png"):
                file_path = os.path.join(folder, file)
                image = cv2.imread(file_path)
                image_shape = str(np.array(image).shape)

                if (image_shape in dict_res.keys()):
                    dict_res[image_shape] += 1
                else:
                    dict_res[image_shape] = 1
                image_count += 1
        print("已处理文件夹{}个，仍需处理文件夹{}个".format(index, len(folder_list) - index))
    print(dict_res)
    print(dict_num)

    dict_num = myf.sort_dict(dict_num, length=20)

    # 绘制柱状图
    figure_count = 0
    myf.draw_bars(dict_res, figure_count + 1, "resolution", "number", "number of different resolution of dataset")
    figure_count += 1
    myf.draw_bars(dict_num, figure_count + 1, "image number", "number", "")
    plt.show()

    # width = 1./len(x_name)
    # for i in range(len(x)):
    #     x[i] = x[i]*width + 0.1
    # bars1 = plt.bar(x, y, width, tick_label = x_name, fc = 'r')
    # # bars2 = plt.bar(x_name, y, width, tick_label = x, fc = 'b')
    #
    # auto_label(bars1)
    # # auto_label(bars2)
    # plt.xlabel("resolution")
    # plt.ylabel("number")
    # plt.title("number of different resolution of dataset")
    # # plt.legend()
    # plt.show()

