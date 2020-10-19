# -*- coding: utf-8 -*-

import os
import platform
import shutil


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
