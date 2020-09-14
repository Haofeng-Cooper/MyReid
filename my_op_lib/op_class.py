# -*- coding: utf-8 -*-
import json
import os
from .op_model import *


class OpArgs():
    def __init__(self, args_flag, args_value):
        self.args_flag = args_flag
        self.args_value = args_value

    def __str__(self):
        return " " + self.args_flag + " " + self.args_value


class OpKeyPoints():
    def __init__(self, model):
        self.version = ""                         # version = 1.3
        self.people = []
        self.pose_body_part_mapping = get_pose_part_mapping(model)
        self.pose_number_body_parts = get_pose_part_number(model)
        self.pose_part_pairs = get_pose_part_pairs(model)


    def load_data_from_file(self, filePath):
        if not os.path.exists(filePath):
            raise IOError("文件不存在： " + filePath)
        with open(filePath, "r") as file:
            json_object = json.load(file)
        self.version = json_object["version"]
        people_list = json_object["people"]
        self.people = []
        for person in people_list:
            person_str = json.dumps(person)
            self.people.append(OpPerson(person_str))



class OpPerson():
    """
    OpKeypoints.people数组中每个元素的格式
    """
    def __init__(self, json_str):
        obj = json.loads(json_str)
        self.person_id = obj["person_id"]
        self.pose_keypoints_2d = obj["pose_keypoints_2d"]
        self.face_keypoints_2d = obj["face_keypoints_2d"]
        self.hand_left_keypoints_2d = obj["hand_left_keypoints_2d"]
        self.hand_right_keypoints_2d = obj["hand_right_keypoints_2d"]
        self.pose_keypoints_3d = obj["pose_keypoints_3d"]
        self.face_keypoints_3d = obj["face_keypoints_3d"]
        self.hand_left_keypoints_3d = obj["hand_left_keypoints_3d"]
        self.hand_right_keypoints_3d = obj["hand_right_keypoints_3d"]



class OpRenderParams():
    def __init__(self):
        # 渲染姿态关键点时使用的参数，数值设置与OpenPose的C++源代码里面一样
        self.thickness_circle_ratio = 1. / 75.
        self.thickness_line_ratio_wrt_circle = 0.75
        self.render_threshold = 0.1
        self.line_type = 8
        self.shift = 0
        self.threshold_rectangle = 0.1
