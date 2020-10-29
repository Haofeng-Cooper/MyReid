# -*- coding: utf-8 -*-
# 数据加载器
import math
import os.path as osp
from glob import glob
import cv2
import numpy as np
from myutils.func import print_process, list_sublist
from myutils.listutils import list_choice


def dataset_reader2(image_folder: str, save_file: str, file_pattern="*.png", force_re=False):
    """
    将目标文件夹下的图像文件数据读取出来，返回 numpy-ndarray 类型的所有图像。
    要求文件命名格式为"相机_行人_图像.png"，且各索引都从0开始。

    同时将读到的图像数据数组写入到文件中，在下次读取时不用循环读取每个图像文件，只需要读取该文件即可。
    
    优先使用 save_file 而不是 image_folder。

    :param image_folder: 存储图像数据的文件夹路径
    :param save_file: 读取到的图像数据要存储的位置，需要 .npz 的后缀名
    :param file_pattern: 在folder下读取什么样名字的文件，如 *.png 表示读取所有.png文件
    :param force_re: 是否重新读取 image_folder 下的所有文件并生成数据文件 save_file
    """
    if not force_re and osp.isfile(save_file):
        # 数据已存在文件中
        print("使用数据文件：" + save_file)
        result = np.load(save_file)
        # image_data, seq_range, cam_offset = result["arr_0"], result["arr_1"], result["arr_2"]
        return result["arr_0"], result["arr_1"], result["arr_2"]

    if not osp.isdir(image_folder):
        raise ValueError("目标文件夹不存在！ " + image_folder)

    print("读取目标目录{}下的文件".format(image_folder))
    files_list = list(sorted(glob(osp.join(image_folder, file_pattern))))
    image_num = len(files_list)
    print("共有{}个要读取的文件".format(image_num))

    image_array = []
    seq_range_indices = []                               # 每个相机下每个行人图像的序列在 image_array 中的起始和终止下标，长度为所有相机下的视频数总和
    cam_offset_indices = [0]                             # 依次保存每个相机下的0号行人在 seq_range_indices 中的下标偏移量
    last_person_id, last_image_id = -1, 100
    seq_start_index, index = 0, 0
    for index, filename in enumerate(files_list):
        print_process(index+1, image_num)
        name_list = osp.basename(filename).split('.')[0].split('_')
        cam_id, person_id, image_id = int(name_list[0]), int(name_list[1]), int(name_list[2])

        if image_id < last_image_id:
            # 表明上一个行人的图像序列结束了
            seq_range_indices.append([last_person_id, seq_start_index, index])
            seq_start_index = index                           # 更新下一个序列的起始下标

        if person_id < last_person_id:
            # 表明上一个相机下的所有图像序列都结束了
            cam_offset_indices.append(len(seq_range_indices))
        last_person_id, last_image_id = person_id, image_id
        image_array.append(cv2.imread(filename))
    seq_range_indices.append([last_person_id, seq_start_index, index])

    # 去除开头多加的一项 [-1, 0, 0]
    seq_range_indices.pop(0)
    cam_offset_indices[1] -= 1

    # 打印相关信息
    print("各相机下的行人为：" + str(cam_offset_indices))
    print("各相机下各行人的视频帧的起始下标和终止下标依次为：")
    print("%-10s|%-10s|%-10s|%-10s" % ("index", "person", "start", "stop"))
    for index, info in enumerate(seq_range_indices):
        print("%-10d|%-10d|%-10d|%-10d" % (index, info[0], info[1], info[2]))

    # 将信息转为 numpy-ndarray 并写入到文件
    image_array = np.asarray(image_array)
    seq_range_indices = np.asarray(seq_range_indices)
    cam_offset_indices = np.asarray(cam_offset_indices)
    np.savez(save_file, image_array, seq_range_indices, cam_offset_indices)

    return image_array, seq_range_indices, cam_offset_indices


def train_data_loader(train_ids, probe_gallery_data, seq_range_, cam_offset_, batch_size=8):
    """
    返回每个 batch 的 anchor, positive, negative 样本数据
    """
    train_num = len(train_ids)
    batch_num = math.ceil(train_num / batch_size)
    result, index = [], 0
    for batch_id in range(batch_num):
        batch_result = []
        for i in range(batch_size):
            anchor_pid = train_ids[index % train_num]
            positive_pid = anchor_pid
            negative_pid = list_choice(train_ids, positive_pid)

            anchor_seq_info = seq_range_[anchor_pid + cam_offset_[0]]
            positive_seq_info = seq_range_[positive_pid + cam_offset_[1]]
            negative_seq_info = seq_range_[negative_pid + cam_offset_[1]]

            # print("计算得：anchor真实行人索引{}，positive行人索引{}，negative行人索引{}".format(
            #     anchor_pid, positive_pid, negative_pid))
            # print("取片段：anchor真实行人索引{}，positive行人索引{}，negative行人索引{}".format(
            #     anchor_seq_info[0], positive_seq_info[0], negative_seq_info[0]))

            anchor = probe_gallery_data[anchor_seq_info[1]: anchor_seq_info[2]]
            positive = probe_gallery_data[positive_seq_info[1]: positive_seq_info[2]]
            negative = probe_gallery_data[negative_seq_info[1]: negative_seq_info[2]]

            batch_result.append([anchor, positive, negative])
            index += 1
        result.append(batch_result)
    return result


def dataset_reader(folder: str):
    """
    将目标文件夹下的图像文件数据读取出来，返回numpy-ndarray的list。
    一维索引为相机索引，二维索引为行人索引，三维索引为图像索引。
    """
    if not osp.isdir(folder):
        raise ValueError("目标文件夹不存在！ "+folder)

    cam_max, person_max = 0, 0
    files_list = list(sorted(glob(osp.join(folder, "*.png"))))
    img_num = len(files_list)
    for filename in files_list:
        names_list = osp.basename(filename).split('.')[0].split('_')
        cam_id, person_id, image_id = int(names_list[0]), int(names_list[1]), int(names_list[2])
        if cam_id >= cam_max:
            cam_max = cam_id + 1
        if person_id >= person_max:
            person_max = person_id + 1
    print("共发现{}张图像，相机数{}，相机下最大人次{}".format(img_num, cam_max, person_max))

    # 读取每一个文件，放到临时列表中
    tmp_lists = [[[] for _ in range(person_max)] for _ in range(cam_max)]
    for index, filename in enumerate(files_list):
        print_process(index+1, img_num)
        names_list = osp.basename(filename).split('.')[0].split('_')
        cam_id, person_id = int(names_list[0]), int(names_list[1])
        tmp_lists[cam_id][person_id].append(cv2.imread(filename))

    # 去除空列表
    images_lists = []
    for cam_id in range(cam_max):
        cam_images = []
        for person_id in range(person_max):
            if len(tmp_lists[cam_id][person_id]) > 0:
                cam_images.append(tmp_lists[cam_id][person_id])
        images_lists.append(cam_images)
    tmp_lists.clear()
    return images_lists


def get_snippet_indices(seq: list, snippet_len=8, stride=3, only_index=False):
    """
    对视频序列进行片段分割
    :param seq: 帧序列
    :param snippet_len: 分割的片段长度
    :param stride: 分割的步长
    :param only_index 是否只获取每个片段的下标
    :return: [[],[],...] 所有分割的下标
    """
    result_indices = []
    seq_len = len(seq)
    if seq_len <= snippet_len:
        snippet_indices = list(range(snippet_len))
        snippet_indices[seq_len:snippet_len] = [seq_len - 1 for _ in range(snippet_len - seq_len)]
        result_indices.append(snippet_indices)
    else:
        snippet_num = (seq_len - snippet_len) // stride + 1
        for i in range(snippet_num):
            snippet_i_indices = list(range(i * stride, i * stride + snippet_len))
            result_indices.append(snippet_i_indices)

    if only_index:
        return result_indices
    else:
        result = []
        for i in range(len(result_indices)):
            result.append(list_sublist(seq, result_indices[i]))
        return result


def snippets_list_concat(snippets_list: list, person_ids):
    """
    对snippets_list进行拼接，(person_num * snippets_num * H * W * 3)，
    其中snippets_num是不相同的。
    person_ids是snippets_list的每一个元素对应的真实行人标签。

    返回连接后的snippets_list,(sum(len(snippets_num) * H * W * 3)
    和每一个元素对应的行人真实索引
    以及该真实行人索引对应在snippets_list中的起始下标
    """
    result_list, label_list, start_index_list = [], [], [0]
    for i, snippets in enumerate(snippets_list):
        result_list += snippets
        label_list += [person_ids[i] for _ in range(len(snippets))]
        start_index_list.append(start_index_list[-1] + len(snippets))
    return result_list, label_list, start_index_list


if __name__ == '__main__':
    # frames = get_frames("D:/datasets/dst/prid_2011/src", 0, 0)
    # print(len(frames))
    # for frame in frames:
    #     print(frame)

    # dataset_reader2("D:/datasets/dst/prid_2011/src")
    # dataset_reader("D:/test/result/src")

    # indices = get_snippet_indices(list(range(100, 120)), only_index=False)
    # indices = get_snippet_indices(list(range(100, 120)))
    # for index_list in indices:
    #     print(index_list)

    image_data, seq_range, cam_offset = dataset_reader2("/home/haofeng/Desktop/datasets/dst/prid_2011/src",
                    save_file="/home/haofeng/Desktop/datasets/dst/prid_2011/src_array.npz")
    print(image_data.shape, seq_range.shape, cam_offset.shape)
    print(image_data.dtype)
    # for i in range(3):
    #     d = image_data[:, :, :, i] / 255
    #     print(np.mean(d), np.std(d))
