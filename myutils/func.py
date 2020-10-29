# -*- coding: utf-8 -*-

import os
import platform
import shutil
import scipy.io as scio
import json
import numpy as np
import random
import cv2
import torch


def positive_round(x):
    """
    四舍五入

    :param x: 浮点数x
    :return: 四舍五入后的整数
    """
    return int(x + 0.5)


def str_not_empty(s: str):
    """
    字符串不为null且不为空时返回True

    :param s: 字符串
    :return: True/False
    """
    if s is not None and len(s) > 0:
        return True
    return False


def clear_folder(folder):
    """
    清空文件夹
    :param folder:
    :return:
    """
    if isinstance(folder, str):
        name_list = os.listdir(folder)
        for name in name_list:
            file_path = os.path.abspath(os.path.join(folder, name))
            os.remove(file_path)
        if len(os.listdir(folder)) != 0:
            raise IOError("文件夹清空失败！{}".format(folder))
    elif isinstance(folder, list):
        for item in folder:
            name_list = os.listdir(item)
            for name in name_list:
                file_path = os.path.abspath(os.path.join(item, name))
                os.remove(file_path)


def mkdir_if_missing(folder_path):
    """
    当文件夹不存在时，创建该文件夹
    :param folder_path: 文件夹路径
    :return: None
    """
    if isinstance(folder_path, list):
        for item in folder_path:
            if not os.path.exists(item):
                os.mkdir(item)
    elif isinstance(folder_path, str):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)


def format_seconds(seconds):
    """
    将秒数转化为天、时、分、秒字符串
    :param seconds:
    :return:
    """
    factor_minute = 60
    factor_hour = factor_minute * 60
    factor_day = factor_hour * 24

    seconds = int(seconds)

    days = seconds // factor_day
    seconds -= days * factor_day

    hours = seconds // factor_hour
    seconds -= hours * factor_hour

    minutes = seconds // factor_minute
    seconds -= minutes * factor_minute

    if days > 0:
        return "{}天{}时{}分{}秒".format(days, hours, minutes, seconds)
    elif hours > 0:
        return "{}时{}分{}秒".format(hours, minutes, seconds)
    elif minutes > 0:
        return "{}分{}秒".format(minutes, seconds)
    else:
        return "{}秒".format(seconds)


def copy_file(src_file, dst_file):
    if platform.system() == "Linux":
        command = ("cp %s %s" % (src_file, dst_file))
        os.system(command)
    else:
        shutil.copy(src_file, dst_file)


def print_process(cur, total, sign_num=20, sign_ch='#'):
    """
    打印进度条
    """
    rate = cur / total
    show_sign_num = max(int(rate * sign_num), 1)
    print('\r进度：%s %.2f%%\t%d/%d' % (sign_ch*show_sign_num, rate*100, cur, total), end='')
    if cur == total:
        print("")


def list_sublist(seq: list, index_list: list):
    """
    获取列表的子元素列表，下标由index_list确定
    :param seq: 数据list
    :param index_list: 要使用的元素在seq中的下标集合
    :return: 采样的片段
    """
    sub = []
    for i in range(len(index_list)):
        sub.append(seq[index_list[i]])
    return sub


def list_shape(_list: list):
    return np.array(_list).shape


def list_concat(_list: list):
    """
    将 _list 的每一个元素进行连接，形成一个新的 result_list。

    _list 的每一个元素除了第一维不同之外，其余各维度都相同。
    除了返回新的 result_list 之外，还要返回一个标记原 _list 中每一个元素在新 result_list 中的起始下标的 index_list。
    index_list 的长度为 _list 的长度+1，原 _list 中元素i对应新的 result_list 中分片为 [ index_list[i]: index_list[i+1] ]
    """
    result_list, index_list = [], [0]
    for snippets in _list:
        result_list = result_list + snippets
        index_list.append(index_list[-1] + len(snippets))

    return result_list, index_list


def rand_element(_list: list):
    """获取 list 中的一个随机元素"""
    return _list[random.randint(0, len(_list))]


def rand_range(start: int, stop: int, exclude: int):
    """在 [start, stop) 范围内生成一个随机数，但是不能等于exclude"""
    rand_value = exclude
    while rand_value == exclude:
        rand_value = random.randrange(start, stop)
    return rand_value


def shuffle_list(_list1: list, _list2: list):
    """
    按顺序取_list2中下标指定的_list1中的元素，形成新的list

    _list1为数据列表, _list2为下标列表；len(_list2) <= len(_list1)
    """
    result = []
    for index in _list2:
        element = _list1[index]
        result.append(element)
    return result


def draw_4d_list(image_data: list, win_name: str, max_cols=4, space=5):
    """绘制4维列表 (n * h * w * c)"""
    images = np.array(image_data)
    n, h, w, c = images.shape

    rows = int(n / max_cols) + 1
    cols = min(n, max_cols)

    new_h = h * rows + (rows - 1) * space
    new_w = w * cols + (cols - 1) * space
    new_image = np.zeros((new_h, new_w, c), dtype=images.dtype)
    new_image[:, :, :] = 255
    for image_index in range(n):
        i = image_index // max_cols
        j = image_index % max_cols
        row_index = i * (h + space)
        col_index = j * (w + space)

        new_image[row_index:row_index+h, col_index:col_index+w, :] = images[image_index]
    cv2.imshow(win_name + str(images.shape), new_image)


def channel_mean_std(data_list: list):
    """
    获取数据的标准值和方差

    data_list: 6维list，依次为 camera, person, image, h, w, 3
    """
    h, w, c = np.asarray(data_list[0][0][0]).shape
    mean, std = [0.0 for _ in range(c)], [0.0 for _ in range(c)]
    sum_x = [0.0 for _ in range(c)]
    sum_x_2 = [0.0 for _ in range(c)]

    image_count = 0
    cam_num = len(data_list)
    for cam_index in range(cam_num):
        person_num = len(data_list[cam_index])
        for person_index in range(person_num):
            image_num = len(data_list[cam_index][person_index])
            for image_index in range(image_num):
                print("\r{}/{}， {}/{}, {}/{}".format(
                    cam_index+1, cam_num, person_index+1, person_num, image_index+1, image_num), end='')

                image = np.asarray(data_list[cam_index][person_index][image_index][:])
                image = image / 255
                for i in range(image.shape[-1]):
                    new_sum_x = sum_x[i] + np.sum(image[:, :, i])
                    if new_sum_x < sum_x[i]:
                        raise ValueError("计算像素值和的时候出现溢出, {} + {} -> {}",
                                         sum_x[i], np.sum(image[:, :, i]), new_sum_x)
                    else:
                        sum_x[i] = new_sum_x

                    new_sum_x_2 = sum_x_2[i] + np.sum(np.power(image[:, :, i], 2.0))
                    if new_sum_x_2 < sum_x_2[i]:
                        raise ValueError("计算像素值平方和的时候出现溢出, {} + {} -> {}",
                                         sum_x_2[i], np.sum(np.power(image[:, :, i], 2.0)), new_sum_x_2)
                    else:
                        sum_x_2[i] = new_sum_x_2

                if image_count + 1 < image_count:
                    raise ValueError("统计图像总数时发生溢出，{} + 1 -> {}", image_count, image_count+1)
                image_count += 1
    pixel_count = image_count * h * w
    for i in range(c):
        mean[i] = sum_x[i] / pixel_count
        std[i] = np.sqrt(sum_x_2[i] / pixel_count - np.power(mean[i], 2.0))
    print("共计{}张图像，{}个像素".format(image_count, pixel_count))
    print("mean={}, std={}".format(mean, std))
    return mean, std


def ndarray_list_2_tensor(array_list: list, transforms, grad=True):
    """
    numpy.ndarray的list转tensor
    """
    length = len(array_list)
    tensor_list = []
    for i in range(length):
        tensor_list.append(transforms(array_list[i]))
    return torch.stack(tensor_list, 0).requires_grad_(grad)


def ndarray4d_2_tensor(ndarray, transforms, grad=True):
    """
    4d的numpy.ndarray转tensor
    """
    n = ndarray.shape[0]
    tensor_list = []
    for i in range(n):
        tensor_list.append(transforms(ndarray[i]))
    return torch.stack(tensor_list, 0).requires_grad_(grad)
