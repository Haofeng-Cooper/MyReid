# -*- coding: utf-8 -*-
# 数据加载器

import os.path as osp
from glob import glob
import cv2


def dataset_reader(folder: str):
    """将目标文件夹下的图像文件数据读取出来，返回numpy-ndarray的list"""
    cam_num, person_num = -1, -1
    files_list = list(sorted(glob(osp.join(folder, "*.png"))))
    for filename in files_list:
        names_list = osp.basename(filename).split('.')[0].split('_')
        cam_id, person_id, image_id = int(names_list[0]), int(names_list[1]), int(names_list[2])
        if cam_id > cam_num:
            cam_num = cam_id + 1
        if person_id > person_num:
            person_num = person_id + 1
    print(cam_num, person_num)


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






if __name__ == '__main__':
    # frames = get_frames("D:/datasets/dst/prid_2011/src", 0, 0)
    # print(len(frames))
    # for frame in frames:
    #     print(frame)

    dataset_reader("D:/datasets/dst/prid_2011/src")


