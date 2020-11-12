import os.path as osp
import matplotlib.pyplot as plt
import numpy as np


def float_list(str_list):
    result = []
    for s in str_list:
        try:
            result.append(float(s))
        except ValueError:
            print("字符串{}不能转变为浮点数！{}".format(s, str_list))
    return result


def beauty_list(str_list):
    result = []
    clear_list = ['', ' ', '\n', '\r']
    for s in str_list:
        if s not in clear_list:
            result.append(s)
    return result


def plot_file(filepath, top_labels):
    top_list = []
    if not osp.isfile(filepath):
        raise ValueError("{}不是一个文件".format(filepath))

    with open(filepath, 'r') as fp:
        while True:
            line_text = fp.readline().strip()
            if not line_text:
                break
            str_list = line_text.split('[')
            if len(str_list) > 1:
                top_str = str_list[-1].split(']')[0]
                # print(top_str)
                top_ks = float_list(beauty_list(top_str.split(' ')))
                top_list.append(top_ks)

    x = list(range(len(top_list)))
    top_list = np.asarray(top_list)
    for i in range(len(top_labels)):
        plt.plot(x, top_list[:, i], label=top_labels[i])
    plt.legend()
    plt.show()


if __name__ == '__main__':
    top_k_labels = ['top-1', 'top-5', 'top-10', 'top-20']
    # plot_file("/home/haofeng/Desktop/saved_models/prid2011_src_split0/log.txt", top_labels=top_k_labels)
    plot_file("/home/haofeng/Desktop/saved_models/prid2011_pose_split0/log.txt", top_labels=top_k_labels)
