# -*- coding: utf-8 -*-
# 数据加载器

import os.path as osp
from glob import glob
import cv2


def dataset_reader(folder: str):
    """将目标文件夹下的图像文件数据读取出来，返回numpy-ndarray的list"""
    cam_max, person_max = 0, 0
    files_list = list(sorted(glob(osp.join(folder, "*.png"))))
    for filename in files_list:
        names_list = osp.basename(filename).split('.')[0].split('_')
        cam_id, person_id, image_id = int(names_list[0]), int(names_list[1]), int(names_list[2])
        # print("doing: ", cam_id, person_id, image_id)
        if cam_id >= cam_max:
            cam_max = cam_id + 1
        if person_id >= person_max:
            person_max = person_id + 1
    print(cam_max, person_max)

    tmp_lists = [[[] for _ in range(person_max)] for _ in range(cam_max)]
    for filename in files_list:
        names_list = osp.basename(filename).split('.')[0].split('_')
        cam_id, person_id = int(names_list[0]), int(names_list[1])
        tmp_lists[cam_id][person_id].append(cv2.imread(filename))
    images_lists = []
    for cam_id in range(cam_max):
        cam_images = []
        for person_id in range(person_max):
            if len(tmp_lists[cam_id][person_id]) > 0:
                cam_images.append(tmp_lists[cam_id][person_id])
        images_lists.append(cam_images)
    print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    for i in range(len(images_lists)):
        print(len(images_lists[i]))


def get_frames(folder: str, person_index: int, cam_index: int):
    """
    获取目标目录下 某行人在某相机下的图像文件路径列表

    base_name = "%05d_%02d_%05d" % (person_index, cam_index, image_index)
    """
    if osp.isdir(folder):
        filename_like = "%05d_%02d_*.png" % (person_index, cam_index)
        print(filename_like)
        frame_paths = list(sorted(glob(osp.join(folder, filename_like))))
        if len(frame_paths) == 0:
            raise ValueError("无法在 {} 下找到行人 {} 在相机 {} 下的图像集！".format(folder, person_index, cam_index))
        return frame_paths
    else:
        raise ValueError("目标文件夹不存在！" + folder)


def __sample_by_indices(seq, index_list: list):
    """
    按下标列表对序列进行采样
    :param seq: 数据list
    :param index_list: 要使用的元素在seq中的下标集合
    :return: 采样的片段
    """
    snippet = []
    for i in range(len(index_list)):
        snippet.append(seq[index_list[i]])
    return snippet


def get_snippet_indices(seq, sni_len=8, stride=3, only_index=False):
    """
    对视频序列进行片段分割
    :param seq: 帧序列
    :param sni_len: 分割的片段长度
    :param stride: 分割的步长
    :param only_index 是否只获取每个片段的下标
    :return: [[],[],...] 所有分割的下标
    """
    result_indices = []
    seq_len = len(seq)
    if seq_len <= sni_len:
        snippet_indices = list(range(sni_len))
        snippet_indices[seq_len:sni_len] = [seq_len-1 for _ in range(sni_len-seq_len)]
        result_indices.append(snippet_indices)
    else:
        snippet_num = (seq_len - sni_len) // stride + 1
        for i in range(snippet_num):
            snippet_i_indices = list(range(i*stride, i*stride+sni_len))
            result_indices.append(snippet_i_indices)

    if only_index:
        return result_indices
    else:
        result = []
        for i in range(len(result_indices)):
            result.append(__sample_by_indices(seq, result_indices[i]))
        return result


if __name__ == '__main__':
    # frames = get_frames("D:/datasets/dst/prid_2011/src", 0, 0)
    # print(len(frames))
    # for frame in frames:
    #     print(frame)

    # dataset_reader("D:/datasets/dst/prid_2011/src")
    # dataset_reader("D:/test/result/src")

    # indices = get_snippet_indices(list(range(100, 120)), only_index=False)
    indices = get_snippet_indices(list(range(100, 120)))
    for index_list in indices:
        print(index_list)
