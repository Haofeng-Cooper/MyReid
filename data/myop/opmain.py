# -*- coding: utf-8 -*-

import os
import os.path as osp
import cv2
import numpy as np
import math

from data.myop.oputils import OpArg, OpRenderParams, OpKeyPoints, construct_rect
from data.myop.opmodel import *
from myutils.func import positive_round, clear_folder, mkdir_if_missing


def process_images(images_folder, json_folder, rendered_folder, op_demo_path):
    """
    运行OpenPose demo程序处理图像，获得生成的关键点文件和渲染的图像文件及对应的文件夹路径

    :param images_folder: 要处理的图像路径
    :param json_folder: 关键点文件保存位置
    :param rendered_folder: 渲染图像保存位置
    :param op_demo_path: OpenPose demo程序的路径
    :return:
    """
    mkdir_if_missing(json_folder)
    mkdir_if_missing(rendered_folder)
    # 参数设置
    op_demo = osp.abspath(op_demo_path)
    op_path = osp.abspath(osp.join(op_demo, "../.."))
    images_folder = osp.abspath(images_folder)

    # 清空关键点文件目录和渲染的图像文件目录
    clear_folder(json_folder)
    clear_folder(rendered_folder)

    # demo程序运行的参数设置
    demo_args = [OpArg("--image_dir", images_folder),
                 OpArg("--write_json", json_folder),
                 OpArg("--write_images", rendered_folder)]
    args_str = ""
    for arg in demo_args:
        args_str += str(arg)

    # 修改当前工作目录为demo程序所在目录
    cwd = os.getcwd()
    os.chdir(op_path)

    # 执行demo程序，生成关键点文件和对应的openpose渲染的带关键点的图像
    command = op_demo + args_str
    print("running: <{}>".format(command))
    os.system(command)
    os.chdir(cwd)


def render_pose(json_file, shape, model=OpModel.BODY_25, show=False):
    """
    根据关键点json文件，渲染行人骨架图（单个图像处理）
    （参考OpenPose C++源代码）

    :param json_file: 关键点文件路径
    :param shape: 原图像尺寸
    :param model: 使用的OpenPose模型
    :param show: 是否展示渲染的图像
    :return: 渲染的图像
    """
    # 读取关键点文件，获取关键点信息
    op_points = OpKeyPoints(model)
    op_points.load_data(json_file)

    # 创建背景图片
    bg_image = cv2.cvtColor(np.zeros((shape[0], shape[1]), dtype=np.uint8), cv2.COLOR_GRAY2BGR)
    width = bg_image.shape[1]
    height = bg_image.shape[0]
    area = width * height

    # 渲染参数
    rp = OpRenderParams()
    # 模型参数
    pairs = get_pose_part_pairs(model)
    scales = get_pose_scales(model)
    colors = get_pose_colors(model)
    num_scales, num_colors = len(scales), len(colors)
    num_points = get_pose_part_number(model)

    # 渲染
    for person_index, person in enumerate(op_points.people):
        person_rect = construct_rect(points=person.pose_keypoints_2d, threshold=rp.threshold_rectangle)
        if person_rect.area() > 0:
            ratio_area = min(1., max(person_rect.w() / float(width), person_rect.h() / float(height)))
            # Size-dependent variable
            thickness_ratio = max(positive_round(math.sqrt(area) * rp.thickness_circle_ratio * ratio_area), 2)
            # cv2.circle()中负的thickness表示绘制实心圆
            thickness_circle = max(1, (thickness_ratio if ratio_area > 0.05 else -1))
            thickness_line = max(1, positive_round(thickness_ratio * rp.thickness_line_ratio_wrt_circle))
            radius = thickness_ratio / 2  # 绘制圆的半径

            # 绘制线（关键点对）
            for k in range(0, len(pairs), 2):
                index1 = pairs[k] * 3  # 第一个点的x坐标在pose_keypoints_2d中的下标
                index2 = pairs[k + 1] * 3  # 第二个点的x坐标在pose_keypoints_2d中的下标
                if person.pose_keypoints_2d[index1 + 2] > rp.render_threshold \
                        and person.pose_keypoints_2d[index2 + 2] > rp.render_threshold:
                    thickness_line_scaled = positive_round(thickness_line * scales[pairs[k + 1] % num_scales])
                    color_index = pairs[k + 1] * 3  # 着色坐标
                    color = (colors[(color_index + 2) % num_colors],
                             colors[(color_index + 1) % num_colors],
                             colors[color_index % num_colors])
                    point1 = (positive_round(person.pose_keypoints_2d[index1]),
                              positive_round(person.pose_keypoints_2d[index1 + 1]))
                    point2 = (positive_round(person.pose_keypoints_2d[index2]),
                              positive_round(person.pose_keypoints_2d[index2 + 1]))
                    cv2.line(bg_image, pt1=point1, pt2=point2, color=color, thickness=thickness_line_scaled,
                             lineType=rp.line_type, shift=rp.shift)
            # 绘制点（关键点）
            for i in range(num_points):
                point_index = i * 3  # 该关键点的x坐标在pose_keypoints_2d中的下标
                if person.pose_keypoints_2d[point_index + 2] > rp.render_threshold:
                    radius_scaled = positive_round(radius * scales[i % num_scales])
                    thickness_circle_scaled = positive_round(thickness_circle * scales[i % num_scales])
                    color_index = i * 3
                    color = (colors[(color_index + 2) % num_colors],
                             colors[(color_index + 1) % num_colors],
                             colors[color_index % num_colors])
                    center = (positive_round(person.pose_keypoints_2d[point_index]),
                              positive_round(person.pose_keypoints_2d[point_index + 1]))
                    cv2.circle(bg_image, center=center, radius=radius_scaled, color=color,
                               thickness=thickness_circle_scaled, lineType=rp.line_type, shift=rp.shift)

    # 渲染完成
    if show:
        cv2.imshow(json_file, bg_image)
        cv2.waitKey(0)
    return bg_image


def draw_all_image(folder_path, max_cols=10, space=5, show=False):
    """
    将folder_path下的包含所有图像绘制在一张大图上。
    folder_path需全为图像

    :param folder_path: 图像目录
    :param max_cols: 每行最多绘制的图像数
    :param space: 图像之间间隔的像素数
    :param show: 是否展示大图
    :return: 排列后的大图
    """
    image_files = os.listdir(folder_path)
    num_files = len(image_files)
    image_sample = cv2.imread(osp.join(folder_path, image_files[0]))
    width, height = image_sample.shape[1], image_sample.shape[0]

    rows = int(num_files / max_cols) + 1
    cols = min(num_files, max_cols)

    new_height = height * rows + (rows - 1) * space
    new_width = width * cols + (cols - 1) * space
    new_image = np.zeros((new_height, new_width, 3), dtype=np.array(image_sample).dtype)
    new_image[:, :, :] = 255  # 设置背景为白色

    # 依次将图片贴到对应的位置
    for index, file_name in enumerate(image_files):
        i = index // max_cols  # 行索引
        j = index % max_cols  # 列索引
        file_path = osp.join(folder_path, file_name)
        row_index = i * (height + space)
        col_index = j * (width + space)

        tmp_image = cv2.imread(file_path)
        new_image[row_index:row_index + height, col_index:col_index + width, :] = tmp_image

    if show:
        cv2.imshow(folder_path, new_image)
        cv2.waitKey(0)
    return new_image


if __name__ == "__main__":
    image_shape = (640, 320)
    test_images = "C:/Users/haofeng/Desktop/openpose_test_images"
    demo_path = "../openpose/bin/OpenPoseDemo.exe"

    json_folder = "C:/Users/haofeng/Desktop/test/json"
    rendered_folder = "C:/Users/haofeng/Desktop/test/rendered"
    pose_folder = "C:/Users/haofeng/Desktop/test/pose"
    big_folder = "C:/Users/haofeng/Desktop/test/big"

    mkdir_if_missing([json_folder, rendered_folder, pose_folder, big_folder])
    clear_folder([json_folder, rendered_folder, pose_folder, big_folder])

    process_images(images_folder=test_images, json_folder=json_folder, rendered_folder=rendered_folder,
                   op_demo_path=demo_path)

    for file_name in os.listdir(json_folder):
        json_path = osp.abspath(osp.join(json_folder, file_name))
        basename = osp.basename(json_path)
        pose_image = render_pose(json_path, image_shape)
        save_path = osp.join(pose_folder, basename+".png")
        cv2.imwrite(save_path, pose_image)

    big_image = draw_all_image(test_images)
    cv2.imwrite(osp.join(big_folder, "all_source.png"), big_image)

    big_image = draw_all_image(rendered_folder)
    cv2.imwrite(osp.join(big_folder, "all_rendered.png"), big_image)

    big_image = draw_all_image(pose_folder)
    cv2.imwrite(osp.join(big_folder, "all_pose.png"), big_image)

    print("over!")
