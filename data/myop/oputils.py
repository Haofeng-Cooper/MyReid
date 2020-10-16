# -*- coding: utf-8 -*-

from data.myop.opmodel import *
from os import path as osp
import json


class OpArg:
    """
    使用OpenPose Demo程序时的命令行参数

    包括参数名和参数取值
    """
    def __init__(self, flag, value):
        """
        初始化
        :param flag: 参数名
        :param value: 参数取值
        """
        self.arg_flag = flag
        self.arg_value = value

    def __str__(self):
        return " {} {}".format(self.arg_flag, self.arg_value)


class OpPerson:
    """关键点json文件中每个行人关键点数据的格式"""
    def __init__(self, json_str):
        obj = json.loads(json_str)
        self.person_id = obj["person_id"]
        self.pose_keypoints_2d = obj["pose_keypoints_2d"]
        # self.face_keypoints_2d = obj["face_keypoints_2d"]
        # self.hand_left_keypoints_2d = obj["hand_left_keypoints_2d"]
        # self.hand_right_keypoints_2d = obj["hand_right_keypoints_2d"]
        # self.pose_keypoints_3d = obj["pose_keypoints_3d"]
        # self.face_keypoints_3d = obj["face_keypoints_3d"]
        # self.hand_left_keypoints_3d = obj["hand_left_keypoints_3d"]
        # self.hand_right_keypoints_3d = obj["hand_right_keypoints_3d"]


class OpKeyPoints:
    """OpenPose身体关键点"""
    def __init__(self, model):
        self.version = ""
        self.people = []
        self.pose_part_number = get_pose_part_number(model)
        self.pose_part_mapping = get_pose_part_mapping(model)
        self.pose_part_pairs = get_pose_part_pairs(model)

    def load_data(self, filepath):
        """
        从保存关键点数据的json文件读取关键点信息
        :param filepath: json文件路径
        :return: none
        """
        if not osp.exists(filepath):
            raise IOError("要加载关键点数据的文件不存在：{}".format(filepath))
        with open(filepath, "r") as f:
            json_object = json.load(f)

            self.people.clear()
            self.version = json_object["version"]

            # 加载每一个人的数据
            people_list = json_object["people"]
            for person in people_list:
                person_str = json.dumps(person)
                self.people.append(OpPerson(person_str))


class OpRenderParams:
    """渲染姿态关键点时使用的参数，数值设置与OpenPose的C++源代码里面一样"""
    def __init__(self):
        self.thickness_circle_ratio = 1. / 75.
        self.thickness_line_ratio_wrt_circle = 0.75
        self.render_threshold = 0.1
        self.line_type = 8
        self.shift = 0
        self.threshold_rectangle = 0.1


class OpRect:
    """矩形框"""

    def __init__(self, x=0, y=0, w=0, h=0):
        self._x = x
        self._y = y
        self._w = w
        self._h = h

    def w(self):
        return self._w

    def h(self):
        return self._h

    def area(self):
        return self._w * self._h


def construct_rect(points: list, threshold):
    """
    构造矩形框

    :param points: 关键点信息数组，长度为关键点数目的3倍。依次为每个关键点的 (x, y, score)
    :param threshold: 置信度阈值，当关键点的score大于此值时认为关键点有效
    :return: MyRect，矩形框
    """
    min_x, min_y, max_x, max_y = 10000, 10000, -1, -1
    for i in range(0, len(points), 3):
        x, y, score = points[i], points[i + 1], points[i + 2]

        if score > threshold:
            min_x, min_y = min(x, min_x), min(y, min_y)
            max_x, max_y = max(x, max_x), max(y, max_y)
    if max_x >= min_x and max_y >= min_y:
        return OpRect(min_x, min_y, max_x - min_x, max_y - min_y)
    else:
        return OpRect()
