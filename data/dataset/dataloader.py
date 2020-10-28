# -*- coding: utf-8 -*-
# 数据加载器

import os.path as osp
from glob import glob
import cv2
from myutils.func import print_process, list_sublist


def dataset_reader(folder: str, max_person=200):
    """
    将目标文件夹下的图像文件数据读取出来，返回numpy-ndarray的list。
    一维索引为相机索引，二维索引为行人索引，三维索引为图像索引。
    """
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
        if person_id >= max_person:
            continue
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

    dataset_reader("D:/datasets/dst/prid_2011/src")
    # dataset_reader("D:/test/result/src")

    # indices = get_snippet_indices(list(range(100, 120)), only_index=False)
    # indices = get_snippet_indices(list(range(100, 120)))
    # for index_list in indices:
    #     print(index_list)
