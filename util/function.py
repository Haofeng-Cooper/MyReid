# -*- coding: utf-8 -*-
import os
import cv2
from matplotlib import pyplot as plt

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
        # os.mkdir(folder_path)
        os.makedirs(folder_path)
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


def string_subtract(str1, str2):
    """
    从字符串str1中减去str2。str2中的字符必须与str1最左边字符依次匹配，且str2长度小于str1长度，才可相减。
    如"abcdf" - "ab" = "cdf"
    :param str1:
    :param str2:
    :return:
    """
    list1 = list(str1)
    list2 = list(str2)
    if len(list2) > len(list1):
        return None
    for i in range(len(list2)):
        if list1[i] != list2[i]:
            return None
    return ''.join( list1[len(list2):] )


def sort_dict(dict_wanted, length=10):
    keys = []
    values = []
    for key in dict_wanted.keys():
        keys.append(int(key))
        values.append(dict_wanted[key])

    print("key max:{}\tkey min:{}".format(max(keys), min(keys)))

    min_key, max_key = min(keys), max(keys)
    grade_num = (max_key - min_key) / length
    if not isinstance(grade_num, int):
        grade_num = int(grade_num) + 1

    range_list = [0] * grade_num
    for i in range(grade_num):
        range_list[i] = min_key + i * length
    range_list.append(range_list[-1] + length)

    number_range = [0] * (len(range_list) - 1)
    for i in range(grade_num):
        low = range_list[i]
        high = range_list[i+1]
        for j in range(low, high, 1):
            if j in keys:
                index = keys.index(j)
                number_range[i] += values[index]
        print("{}\tfrom:{}\tto:{}\tnumber:{}".format(i, low, high, number_range[i]))

    result = {}
    for i in range(grade_num):
        key = "{}~{}".format(range_list[i], range_list[i + 1])
        key = int((range_list[i] + range_list[i + 1]) / 2)
        value = number_range[i]
        if value > 0:
            result[key] = value

    return result


def auto_label(rects):
    """显示柱上的数值"""
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x()+rect.get_width()/2.0, 1.03*height, "{}".format(int(height)))


def draw_bars(dict_data, figure, x_label, y_label, title):
    x_name = []
    y = []
    for key in dict_data.keys():
        x_name.append(str(key))
        y.append(dict_data[key])
    x_index = list(range(len(x_name)))
    # print("x_name: ", x_name)
    # for i in range(1, len(x_name), 2):
    #     x_name[i] = ""
    # print("x_name: ", x_name)

    plt.figure(figure)
    bars = plt.bar(x_index, y, color=list("rgb"), tick_label=x_name)
    # bars = plt.bar(x_index, y, color=list("rgb"))
    auto_label(bars)
    if str_not_empty(x_label):
        plt.xlabel(x_label)
    if str_not_empty(y_label):
        plt.ylabel(y_label)
    if str_not_empty(title):
        plt.title(title)


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



if __name__ == "__main__":
    # print(string_subtract("D:\\reid_datasets\\prid_2011\\multi_shot\\cam_b\\person_0741", "D:\\reid_datasets\\prid_2011\\multi_s"))

    # d = {'142': 4, '117': 11, '136': 6, '163': 5, '135': 16, '5': 25, '176': 1, '149': 5, '97': 14, '116': 12, '158': 3, '151': 4, '131': 9, '255': 1, '119': 9, '111': 8, '73': 12, '36': 5, '25': 6, '98': 9, '90': 9, '155': 3, '152': 2, '129': 5, '110': 10, '34': 5, '120': 6, '112': 13, '127': 6, '7': 15, '170': 1, '260': 1, '87': 8, '141': 4, '122': 8, '144': 4, '138': 7, '172': 2, '125': 6, '126': 8, '161': 3, '108': 10, '139': 3, '132': 3, '146': 3, '118': 11, '21': 5, '95': 5, '156': 5, '88': 10, '134': 8, '109': 9, '82': 6, '105': 6, '137': 3, '147': 3, '115': 6, '184': 1, '104': 10, '62': 5, '45': 7, '113': 11, '121': 12, '91': 14, '128': 3, '71': 8, '72': 10, '61': 8, '99': 9, '86': 11, '96': 13, '83': 9, '58': 7, '174': 1, '74': 8, '168': 1, '162': 3, '102': 5, '106': 10, '31': 6, '94': 12, '175': 2, '166': 2, '160': 3, '107': 7, '26': 6, '16': 9, '77': 3, '6': 15, '357': 1, '8': 11, '123': 6, '103': 16, '114': 6, '11': 10, '100': 9, '78': 5, '222': 1, '84': 10, '133': 7, '171': 2, '675': 1, '85': 10, '165': 3, '64': 9, '101': 12, '17': 9, '39': 8, '41': 12, '28': 8, '182': 1, '81': 12, '124': 2, '67': 6, '51': 12, '9': 14, '56': 11, '231': 1, '93': 8, '154': 1, '235': 1, '520': 1, '79': 8, '194': 1, '207': 1, '180': 1, '66': 12, '140': 3, '279': 1, '14': 7, '159': 3, '92': 9, '12': 8, '264': 1, '284': 1, '230': 1, '435': 1, '296': 1, '65': 2, '191': 1, '18': 3, '181': 1, '53': 5, '35': 7, '130': 5, '252': 1, '455': 1, '153': 1, '33': 5, '22': 7, '195': 1, '63': 7, '47': 5, '75': 6, '43': 5, '80': 3, '59': 6, '54': 4, '179': 1, '38': 5, '69': 7, '49': 7, '13': 4, '30': 4, '70': 6, '55': 4, '40': 6, '29': 6, '89': 4, '57': 6, '50': 8, '68': 6, '76': 7, '60': 7, '27': 4, '10': 7, '19': 8, '32': 2, '20': 7, '37': 2, '23': 4, '48': 8, '46': 3, '173': 1, '52': 2, '150': 1, '157': 1, '148': 1, '42': 6, '204': 1, '15': 3, '44': 2, '24': 3, '290': 1}
    # result = sort_dict(d)
    # print(result)
    # draw_bars(result, 1, "range", "", "")
    # plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    # plt.xticks(fontsize=5)
    # plt.show()

    print(format_seconds(25))
    print(format_seconds(250))
    print(format_seconds(2500))
    print(format_seconds(25000))
    print(format_seconds(250000))
    print(format_seconds(2500000))
    print(format_seconds(25000000))

