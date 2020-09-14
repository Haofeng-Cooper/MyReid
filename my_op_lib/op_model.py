# -*- coding: utf-8 -*-

from enum import Enum
from .params_body_25 import *


class OpModel(Enum):
    BODY_25 = 0
    COCO_18 = 1
    MPI_15 = 2
    ERROR_TEST = 10

# 错误信息提示
MSG_ERROR_MODEL = "模型错误！请使用正确的OpenPose模型！"
MSG_ERROR_MODEL_INDEX = "模型下标越界！请完善或添加模型对应的参数！"

# 关键点映射
POSE_PART_MAPPING_LIST = [
    POSE_BODY_25_PART_MAPPING
]
# 关键点数目
POSE_PART_NUMBER_LIST = [
    POSE_BODY_25_PART_NUMBER
]
# 可绘制的关键点对
POSE_PART_PAIRS_LIST = [
    POSE_BODY_25_PART_PAIRS
]
# 关键点绘制时的缩放尺度
POSE_SCALES_LIST = [
    POSE_BODY_25_SCALES
]
# 关键点着色方案
POSE_COLORS_LIST = [
    POSE_BODY_25_COLORS
]


def get_pose_part_mapping(model):
    """
    获取model对应的身体关键点部位映射表（字典）
    :param model:
    :return:
    """
    # print(type(model_name))
    if isinstance(model, OpModel):
        if model.value < len(POSE_PART_MAPPING_LIST):
            return POSE_PART_MAPPING_LIST[model.value]
        else:
            raise IndexError(MSG_ERROR_MODEL_INDEX)
    else:
        raise ValueError(MSG_ERROR_MODEL)


def get_pose_part_number(model):
    """获取model对应的身体关键点部位数目"""
    if isinstance(model, OpModel):
        if model.value < len(POSE_PART_NUMBER_LIST):
            return POSE_PART_NUMBER_LIST[model.value]
        else:
            raise IndexError(MSG_ERROR_MODEL_INDEX)
    else:
        raise ValueError(MSG_ERROR_MODEL)


def get_pose_part_pairs(model):
    """获取model对应的身体关键点对"""
    if isinstance(model, OpModel):
        if model.value < len(POSE_PART_PAIRS_LIST):
            return POSE_PART_PAIRS_LIST[model.value]
        else:
            raise IndexError(MSG_ERROR_MODEL_INDEX)
    else:
        raise ValueError(MSG_ERROR_MODEL)


def get_pose_scales(model):
    if isinstance(model, OpModel):
        if model.value < len(POSE_SCALES_LIST):
            return POSE_SCALES_LIST[model.value]
        else:
            raise IndexError(MSG_ERROR_MODEL_INDEX)
    else:
        raise ValueError(MSG_ERROR_MODEL)


def get_pose_colors(model):
    if isinstance(model, OpModel):
        if model.value < len(POSE_COLORS_LIST):
            return POSE_COLORS_LIST[model.value]
        else:
            raise IndexError(MSG_ERROR_MODEL_INDEX)
    else:
        raise ValueError(MSG_ERROR_MODEL)



if __name__ == "__main__":
    model = OpModel.BODY_25
    print(get_pose_part_mapping(model))
    print(get_pose_part_number(model))
    print(get_pose_part_pairs(model))
    print(get_pose_scales(model))
    print(get_pose_colors(model))

    print(get_pose_part_pairs(OpModel.ERROR_TEST))
