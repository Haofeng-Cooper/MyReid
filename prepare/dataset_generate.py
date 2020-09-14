# -*- coding: utf-8 -*-

import cv2
import shutil
from util.file_util import *
import util.function as myf
from queue import Queue, LifoQueue
from my_op_lib.op_main import OpMain
import time


def is_wanted_folder(fileFolder):
    if isinstance(fileFolder, FileFolder):
        if len(fileFolder.child_files) > 0 and len(fileFolder.child_folders) == 0:
            flag = True
            for file in fileFolder.child_files:
                if not myf.str_contains(file, ".png"):
                    flag = False
                    break
            return flag
    return False


def main(source_dir, target_dir, image_shape):
    target_images_dir = os.path.join(target_dir, "images")
    target_keypoint_dir = os.path.join(target_dir, "keypoint")
    target_rendered_dir = os.path.join(target_dir, "rendered")
    target_pose_dir = os.path.join(target_dir, "pose")

    # 参数设定
    openpose_path = "../my_op_lib/openpose"
    max_cols = 12
    space = 5
    tmp_folder = "./tmp"
    source_image_suffix = ".png"
    keypoints_file_suffix = "_keypoints.json"
    rendered_image_suffic = "_rendered.png"
    skeleton_image_suffix = "_skeleton.png"


    # 获取包含图像文件的文件夹列表
    folder_queue = LifoQueue()
    folder_list = []
    if os.path.isdir(source_dir):
        root = FileFolder(source_dir)
        folder_queue.put(root)
        while not folder_queue.empty():
            cur = folder_queue.get()
            if is_wanted_folder(cur):
                folder_list.append(cur.path.replace("\\", "/"))
            size = len(cur.child_folders)

            for index in range(size):
                child_folder_path = cur.child_folders[size-1-index]
                child_folder = FileFolder(child_folder_path, father=cur, FILE_FLAG=FLAG_FOLDER)
                folder_queue.put(child_folder)

    # 获取openpose的demo程序
    demo = OpMain()

    # 逐个处理每个文件夹
    for index, folder in enumerate(folder_list):
        cur_time = time.time()

        # 根据要处理的文件的路径构造要保存的文件所在路径
        relative_path = myf.string_subtract(folder, source_dir)[1:]
        images_path = os.path.join(target_images_dir, relative_path).replace("\\", "/")
        keypoint_path = os.path.join(target_keypoint_dir, relative_path).replace("\\", "/")
        rendered_path = os.path.join(target_rendered_dir, relative_path).replace("\\", "/")
        pose_path = os.path.join(target_pose_dir, relative_path).replace("\\", "/")

        print("{}\t{}\t{}\t{}\t{}\t{}".format(folder, relative_path, images_path, keypoint_path, rendered_path, pose_path))

        myf.clear_folder(images_path, force_clear=True)
        myf.clear_folder(keypoint_path, force_clear=True)
        myf.clear_folder(rendered_path, force_clear=True)
        myf.clear_folder(pose_path, force_clear=True)

        # 0 将源目录下的源文件拷贝到目标目录下 folder -> images_path
        source_files = myf.list_contain_str(os.listdir(folder), source_image_suffix)
        for file in source_files:
            src_path = os.path.join(folder, file)
            dst_path = os.path.join(images_path, file)
            shutil.copy(src_path, dst_path)

        # 1 生成关键点文件和渲染了姿态的图像文件
        result_keypoints = demo.process_images(images_to_process_path=images_path,
                                               keypoints_files_path=keypoint_path,
                                               rendered_image_path=rendered_path,
                                               openpose_path=openpose_path)
        # 得到关键点文件
        keypoints_files_folder = result_keypoints["keypoints_files_folder"]
        keypoints_files = result_keypoints["keypoints_files"]

        # 2 根据关键点文件绘制背景为黑色的骨架图，并保存到指定目录下
        for file_name in keypoints_files:
            if not file_name.endswith(keypoints_file_suffix):
                raise ValueError("关键点文件{}没有以指定后缀{}结尾！".format(file_name, keypoints_file_suffix))
            src_name = file_name[:file_name.find(keypoints_file_suffix)]
            image = demo.render_pose(os.path.join(keypoints_files_folder, file_name),
                                     src_image_shape=image_shape)
            save_path = os.path.join(pose_path, src_name + skeleton_image_suffix)
            cv2.imwrite(save_path, image)

        print("完成度{}/{},预计仍需耗时{}".format(index+1, len(folder_list),
                                         myf.format_seconds((len(folder_list)-1-index) * (time.time()-cur_time))))

if __name__ == "__main__":
    source_dir = "D:/reid_datasets/prid_2011/multi_shot"
    target_dir = "D:/result_datasets/prid_2011"
    image_shape = (128, 64)

    main(source_dir, target_dir, image_shape)
