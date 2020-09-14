# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import math
from my_op_lib import function as myf
from my_op_lib import op_class as opc
from my_op_lib import op_function as opf
from my_op_lib import op_model as opm

class OpMain():
    def __init__(self):
        # 临时文件目录
        self._tmp_path = ".\\tmp"
        # windows下gpu版openpose路径，已包含模型文件
        self.openpose_path = ".\\openpose"
        # demo程序的路径
        self.openpose_demo_path = os.path.join(self.openpose_path, "bin\\OpenPoseDemo.exe")
        # 要处理的图像文件所在路径
        self.images_to_process_path = ""
        # 处理图像生成的关键点文件应该存放的目录
        self.keypoints_files_path = self._tmp_path
        # 根据关键点在原图上绘制完骨架后生成的图像应该存放的目录
        self.rendered_images_path = self._tmp_path

    def process_images(self, images_to_process_path=None, keypoints_files_path=None,
                       rendered_image_path=None, openpose_path=None):
        """运行demo程序，对图像进行处理，获得生成的关键点文件和渲染的图像文件以及对应的文件夹路径"""
        if myf.str_not_empty(openpose_path):
            self.openpose_path = openpose_path
            self.openpose_demo_path = os.path.join(self.openpose_path, "bin\\OpenPoseDemo.exe")

        if myf.str_not_empty(images_to_process_path):
            self.images_to_process_path = images_to_process_path
        else:
            raise ValueError("请指定要处理的图像所在文件夹路径！该路径不可缺少也不可为空字符串！")

        if myf.str_not_empty(keypoints_files_path):
            self.keypoints_files_path = keypoints_files_path

        if myf.str_not_empty(rendered_image_path):
            self.rendered_images_path = rendered_image_path

        # # 如果关键点文件路径或渲染的图像路径是self._tmp_path，那么对临时文件进行清空处理
        # if self.keypoints_files_path == self._tmp_path or self.rendered_images_path == self._tmp_path:
        #     myf.clear_folder(self._tmp_path, force_clear=True)
        myf.clear_folder(keypoints_files_path)
        myf.clear_folder(rendered_image_path)

        # 转变成绝对路径
        self.openpose_path = os.path.abspath(self.openpose_path)
        self.openpose_demo_path = os.path.abspath(self.openpose_demo_path)
        self.images_to_process_path = os.path.abspath(self.images_to_process_path)
        self.keypoints_files_path = os.path.abspath(self.keypoints_files_path)
        self.rendered_images_path = os.path.abspath(self.rendered_images_path)


        # 记录关键点文件目录和渲染的图像目录下原有的文件列表
        folder1_files_before = os.listdir(self.keypoints_files_path)
        folder2_files_before = os.listdir(self.rendered_images_path)

        # demo程序执行时附带的参数设置
        demo_args_list = []
        demo_args_list.append(opc.OpArgs("--image_dir", self.images_to_process_path))
        demo_args_list.append(opc.OpArgs("--write_json", self.keypoints_files_path))
        demo_args_list.append(opc.OpArgs("--write_images", self.rendered_images_path))
        # demo_args_list.append(opc.OpArgs("--write_images_format", "png"))

        # 组合所有参数
        args_str = ""
        for args in demo_args_list:
            args_str += str(args)
        print("demo's args:\t" + args_str)

        # 修改当前工作目录
        cwd = os.getcwd()
        os.chdir(self.openpose_path)

        # 执行demo程序
        print(self.openpose_demo_path + args_str)
        os.system(self.openpose_demo_path + args_str)
        os.chdir(cwd)

        # 记录关键点文件目录和渲染的图像目录下现有的文件列表
        folder1_files_after = os.listdir(self.keypoints_files_path)
        folder2_files_after = os.listdir(self.rendered_images_path)

        # 记录新生成的文件并返回
        folder1_diff_files = myf.list_diff(folder1_files_after, folder1_files_before, myf.NEW_FILE_KEYWORD[0])
        folder2_diff_files = myf.list_diff(folder2_files_after, folder2_files_before, myf.NEW_FILE_KEYWORD[1])

        return_dict = {}
        return_dict["keypoints_files_folder"] = self.keypoints_files_path
        return_dict["keypoints_files"] = folder1_diff_files
        return_dict["rendered_images_folder"] = self.rendered_images_path
        return_dict["rendered_images"] = folder2_diff_files

        return return_dict

    def render_pose(self, keypoints_json_file_path, src_image_shape, model=opm.OpModel.BODY_25, show_flag = False):
        """
        根据关键点json文件，渲染关键点姿态图
        """
        # 读取关键点文件，获取关键点信息
        op_keypoints = opc.OpKeyPoints(model)
        op_keypoints.load_data_from_file(keypoints_json_file_path)

        # 创建背景图片
        image = cv2.cvtColor(np.zeros((src_image_shape[0], src_image_shape[1]), dtype=np.uint8), cv2.COLOR_GRAY2BGR)
        width = image.shape[1]
        height = image.shape[0]
        area = width * height

        # 渲染（绘制）参数
        render_params = opc.OpRenderParams()
        thickness_circle_ratio = render_params.thickness_circle_ratio
        thickness_line_ratio_wrt_circle = render_params.thickness_line_ratio_wrt_circle
        render_threshold = render_params.render_threshold
        line_type = render_params.line_type
        shift = render_params.shift
        threshold_rectangle = render_params.threshold_rectangle

        # 模型参数
        pairs = opm.get_pose_part_pairs(model)
        pose_scales = opm.get_pose_scales(model)
        number_scales = len(pose_scales)
        pose_colors = opm.get_pose_colors(model)
        number_colors = len(pose_colors)
        number_keypoints = opm.get_pose_part_number(model)

        # 关键点
        for person_index, person in enumerate(op_keypoints.people):
            person_rectangle = opf.get_keypoints_rectangle(person.pose_keypoints_2d, threshold_rectangle)
            if opf.area(person_rectangle) > 0:
                ratio_area = min(1., max(opf.rectangle_width(person_rectangle) / float(width),
                                         opf.rectangle_height(person_rectangle) / float(height)))
                # Size-dependent variable
                thickness_ratio = max(opf.positive_int_round(math.sqrt(area)
                                                             * thickness_circle_ratio
                                                             * ratio_area),
                                      2)
                # cv2.circle()中负的thickness表示要绘制一个实心圆
                thickness_circle = max(1, (thickness_ratio if ratio_area > 0.05 else -1))
                thickness_line = max(1, opf.positive_int_round(thickness_ratio * thickness_line_ratio_wrt_circle))
                radius = thickness_ratio / 2              # 绘制圆的半径

                # 绘制线（关键点对）
                for k in range(0, len(pairs), 2):
                    index1 = pairs[k] * 3              # 第一个点的x坐标在pose_keypoints_2d中的下标
                    index2 = pairs[k+1] * 3            # 第二个点的x坐标在pose_keypoints_2d中的下标
                    if person.pose_keypoints_2d[index1+2] > render_threshold \
                            and person.pose_keypoints_2d[index2+2] > render_threshold:
                        thickness_line_scaled = opf.positive_int_round(thickness_line * pose_scales[pairs[k+1]%number_scales])
                        color_index = pairs[k+1] * 3
                        color = (pose_colors[(color_index+2) % number_colors],
                                 pose_colors[(color_index+1) % number_colors],
                                 pose_colors[(color_index) % number_colors])
                        point1 = (opf.positive_int_round(person.pose_keypoints_2d[index1]),
                                  opf.positive_int_round(person.pose_keypoints_2d[index1+1]))
                        point2 = (opf.positive_int_round(person.pose_keypoints_2d[index2]),
                                  opf.positive_int_round(person.pose_keypoints_2d[index2+1]))
                        cv2.line(image, point1, point2, color, thickness_line_scaled, line_type, shift)
                # 绘制点（关键点）
                for i in range(number_keypoints):
                    point_index = i * 3               # 该关键点的x坐标在pose_keypoints_2d中的下标
                    if person.pose_keypoints_2d[point_index + 2] > render_threshold:
                        radius_scaled = opf.positive_int_round(radius * pose_scales[i % number_scales])
                        thickness_circle_scaled = opf.positive_int_round(thickness_circle * pose_scales[i % number_scales])
                        color_index = i * 3
                        color = (pose_colors[(color_index+2) % number_colors],
                                 pose_colors[(color_index+1) % number_colors],
                                 pose_colors[(color_index) % number_colors])
                        center = (opf.positive_int_round(person.pose_keypoints_2d[point_index]),
                                  opf.positive_int_round(person.pose_keypoints_2d[point_index + 1]))
                        cv2.circle(image, center, radius_scaled, color, thickness_circle_scaled, line_type, shift)
        if show_flag:
            cv2.imshow(keypoints_json_file_path, image)
            cv2.waitKey(0)
        return image

    def draw_images(self, folder_path, image_keyword, max_cols, space=10):
        """
        将文件夹 folder_path 下的所有指定格式 image_keyword 图像在一张大图上显示出来，一行最多显示max_cols个图像。
        所有要绘制的图像的size是一致的

        :param folder_path:
        :param image_keyword:
        :param max_cols:
        :return: 排列后的大图
        """
        files = myf.list_contain_str(os.listdir(folder_path), image_keyword)
        number_files = len(files)
        image_sample = cv2.imread(os.path.join(folder_path, files[0]))
        height = image_sample.shape[0]
        width = image_sample.shape[1]

        rows = int(number_files/max_cols) + 1
        cols = min(number_files, max_cols)

        new_height = height * rows + (rows - 1) * space
        new_width = width * cols + (cols - 1) * space
        new_image = np.zeros((new_height, new_width, 3), dtype=np.array(image_sample).dtype)
        # cv2.imshow("0", new_image)
        # 设置背景为白色
        new_image[:,:,:] = 255
        # cv2.imshow("255", new_image)
        # cv2.waitKey(0)

        # 依次将图片贴到对应的位置
        for file_index, file_name in enumerate(files):
                i = int(file_index / max_cols)
                j = file_index % max_cols
                file_path = os.path.join(folder_path, file_name)
                row_index = i * height + i * space
                col_index = j * width + j * space
                # print(i, j, row_index, col_index)

                tmp_image = cv2.imread(file_path)
                new_image[row_index:row_index+height, col_index:col_index+width, :] = tmp_image

        return new_image






if __name__ == "__main__":
    # image_shape = (640, 320)
    # test_image_folder = "../test_images"
    image_shape = (128,64)
    test_image_folder = "D:\\reid_datasets\\prid_2011\\multi_shot\\cam_a\\person_0001"
    # test_image_folder = "D:\\reid_datasets\\iLIDS-VID\\i-LIDS-VID\\sequences\\cam1\\person001"
    max_cols = 12
    space = 5
    tmp_folder = "./tmp"
    source_image_suffix = ".png"
    keypoints_file_suffix = "_keypoints.json"
    rendered_image_suffic = "_rendered.png"
    skeleton_image_suffix = "_skeleton.png"

    # 1 生成关键点文件和渲染了姿态的图像文件
    demo = OpMain()
    result_keypoints = demo.process_images(images_to_process_path=test_image_folder,
                                           keypoints_files_path=tmp_folder,
                                           rendered_image_path=tmp_folder)
    keypoints_files_folder = result_keypoints["keypoints_files_folder"]
    keypoints_files = result_keypoints["keypoints_files"]
    rendered_images_folder = result_keypoints["rendered_images_folder"]
    rendered_images = result_keypoints["rendered_images"]

    # 2 根据关键点文件绘制背景为黑色的骨架图，并保存到指定目录下
    for file_name in keypoints_files:
        if not file_name.endswith(keypoints_file_suffix):
            raise ValueError("关键点文件{}没有以指定后缀{}结尾！".format(file_name, keypoints_file_suffix))
        src_name = file_name[:file_name.find(keypoints_file_suffix)]
        image = demo.render_pose(os.path.join(keypoints_files_folder, file_name),
                                 src_image_shape=image_shape)
        save_path = os.path.join(tmp_folder, src_name+skeleton_image_suffix)
        cv2.imwrite(save_path, image)
        # print("save file: " + save_path)

    # 3 将所有骨架图放在同一张大图上
    big_image = demo.draw_images(tmp_folder, skeleton_image_suffix, max_cols, space)
    cv2.imwrite("./show_all_skeleton.png", big_image)

    big_image = demo.draw_images(tmp_folder, rendered_image_suffic, max_cols, space)
    cv2.imwrite("./show_all_rendered.png", big_image)

    big_image = demo.draw_images(test_image_folder, source_image_suffix, max_cols, space)
    cv2.imwrite("./show_all_source.png", big_image)

    print("over!")

