# -*- coding: utf-8 -*-
import os
import cv2

NEW_FILE_KEYWORD = ["keypoints", "rendered", "skeleton"]

def str_not_empty(str):
    """
    当str不为None且不为空字符串时返回true，否则返回false

    :param str:
    :return:
    """
    if str and len(str) > 0:
        return True
    else:
        return False

def str_contains(str1, str2):
    """
    判断str1内是否包含str2

    :param str1:
    :param str2:
    :return:
    """
    if str1 == None:
        return False
    elif str2 == None:
        return True
    elif len(str1) < len(str2):
        return False
    else:
        for i in range(len(str1)):
            flag = True
            for j in range(len(str2)):
                if str1[i+j] != str2[j]:
                    flag = False
                    break
            if flag:
                return True
        return False


def clear_folder(folder_path, cleared_names=None, force_clear=False):
    """
    清空一个文件夹（但只清除名字中含有列表cleared_names中的字符串元素的文件）。
    如果文件夹不存在，则创建它。

    :param folder_path: 目标文件夹路径
    :param cleared_names: 默认['keypoints', 'rendered']
    :param force_clear: 如果为True，则无视clear_names，且清空整个文件夹。默认为false。
    :return:
    """
    if cleared_names is None:
        cleared_names = NEW_FILE_KEYWORD

    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    else:
        file_names = os.listdir(folder_path)
        if force_clear:
            for file_name in file_names:
                file_path = os.path.join(folder_path, file_name)
                os.remove(file_path)
        else:
            for file_name in file_names:
                for cleared_name in cleared_names:
                    if str_contains(file_name, cleared_name):
                        file_path = os.path.join(folder_path, file_name)
                        os.remove(file_path)
                        break


def list_diff(list1, list2, contain_keyword=None):
    """
    求list1中不在list2中的元素列表(可以有重复的)。
    使用该函数的时候，很可能 len(list1) > len(list2)。
    如果指定了contain_keyword，则这些元素还需要包含contain_keyword指定的字符串。

    :param list1:
    :param list2:
    :param contain_keyword:
    :return:
    """
    result = []
    for element in list1:
        if element not in list2 and str_contains(element, contain_keyword):
            result.append(element)
    return result


def list_contain_str(list, str):
    """
    求字符串列表list中包含指定字符串str的元素列表。

    :param list:
    :param str:
    :return:
    """
    result = []
    for element in list:
        if str_contains(element, str):
            result.append(element)
    return result


def get_src_name_from_keypoints_name(file_path, keyword="_keypoints.json"):
    """
    从file_path指定的关键点json文件的文件名中获取对应的原文件名
    :param file_path:
    :param keyword:
    :return:
    """
    base_name = os.path.basename(file_path)
    index = base_name.find(keyword)
    return base_name[:index]


if __name__ == "__main__":
    print(str_contains("0001_keypoints.json", None))
    print(get_src_name_from_keypoints_name("0001_keypoints.json"))



