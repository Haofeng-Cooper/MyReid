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


def get_snippet_indices(seq_len, sni_len=8, stride=3):
    """
    对视频序列进行片段分割
    :param seq_len: 序列长度
    :param sni_len: 分割的片段长度
    :param stride: 分割的步长
    :return: [[],[],...] 所有分割的下标
    """
    result = []
    if seq_len <= sni_len:
        snippet = list(range(sni_len))
        snippet[seq_len:sni_len] = [seq_len-1 for _ in range(sni_len-seq_len)]
        result.append(snippet)
    else:
        snippet_num = (seq_len - sni_len) // stride + 1
        for i in range(snippet_num):
            snippet_i = list(range(i*stride, i*stride+sni_len))
            result.append(snippet_i)
    return result


if __name__ == '__main__':
    # frames = get_frames("D:/datasets/dst/prid_2011/src", 0, 0)
    # print(len(frames))
    # for frame in frames:
    #     print(frame)

    # dataset_reader("D:/datasets/dst/prid_2011/src")
    # dataset_reader("D:/test/result/src")

    indices = get_snippet_indices(15)
    for index_list in indices:
        print(index_list)
    print(indices)
