# -*- coding: utf-8 -*-

import os
import os.path as osp
import cv2
import time

from myutils.func import mkdir_if_missing, clear_folder, copy_file
from data.myop.opmain import process_images, render_pose


class MyDataSet:
    def __init__(self, dataset_path, op_demo_path):
        self.root_path = dataset_path
        self.folder_list = []
        self.op_demo = op_demo_path

    def fill_folder_list(self):
        self.folder_list.clear()
        cam_list = os.listdir(self.root_path)
        for cam_name in cam_list:
            path_list = []
            cam_path = osp.join(self.root_path, cam_name)
            person_list = os.listdir(cam_path)
            for person_name in person_list:
                person_path = osp.join(cam_path, person_name)
                path_list.append(person_path)
            self.folder_list.append(path_list)

    def process(self, tmp_folder, dst_folder, image_shape, image_format=".png"):
        tmp_folder_rendered = osp.join(tmp_folder, "rendered")
        tmp_folder_json = osp.join(tmp_folder, "json")
        mkdir_if_missing(tmp_folder_rendered)
        mkdir_if_missing(tmp_folder_json)
        clear_folder([tmp_folder_rendered, tmp_folder_json])

        dst_folder_src = osp.join(dst_folder, "src")
        dst_folder_rendered = osp.join(dst_folder, "rendered")
        dst_folder_json = osp.join(dst_folder, "json")
        dst_folder_pose = osp.join(dst_folder, "pose")
        mkdir_if_missing(dst_folder_src)
        mkdir_if_missing(dst_folder_rendered)
        mkdir_if_missing(dst_folder_json)
        mkdir_if_missing(dst_folder_pose)
        clear_folder([dst_folder_src, dst_folder_rendered, dst_folder_json, dst_folder_pose])

        for cam_index, person_list in enumerate(self.folder_list):
            for person_index, person_folder in enumerate(person_list):
                process_images(person_folder, tmp_folder_json, tmp_folder_rendered, self.op_demo)
                rendered_list = os.listdir(tmp_folder_rendered)
                json_list = os.listdir(tmp_folder_json)
                image_list = os.listdir(person_folder)
                for image_index, image_name in enumerate(image_list):
                    # 生成pose并保存
                    base_name = "%05d_%02d_%05d" % (person_index, cam_index, image_index)
                    json_path = osp.join(tmp_folder_json, json_list[image_index])
                    pose_image = render_pose(json_path, image_shape)
                    save_path = osp.join(dst_folder_pose, base_name+image_format)
                    cv2.imwrite(save_path, pose_image)

                    # 拷贝源图像文件到目标类目录下并重命名
                    copy_file(osp.join(person_folder, image_name),
                              osp.join(dst_folder_src, base_name+image_format))
                    # 拷贝tmp文件到dst
                    copy_file(osp.join(tmp_folder_rendered, rendered_list[image_index]),
                              osp.join(dst_folder_rendered, base_name+image_format))
                    copy_file(json_path, osp.join(dst_folder_json, base_name+".json"))
                clear_folder([tmp_folder_rendered, tmp_folder_json])
                print("%02d %05d finished" % (cam_index, person_index))
