# -*- coding: utf-8 -*-


from enum import Enum
from data.myop.model_params.body_25 import *


class OpModel(Enum):
    """
    OpenPose模型
    """
    BODY_25 = 0
    COCO_18 = 1
    MPI_15 = 2


# 以模型为索引下标的
# 关键点数目
POSE_PART_NUMBER_LIST = [
    POSE_BODY_25_PART_NUMBER
]
# 关键点映射
POSE_PART_MAPPING_LIST = [
    POSE_BODY_25_PART_MAPPING
]
# 关键点对
POSE_PART_PAIRS_LIST = [
    POSE_BODY_25_PART_PAIRS
]
# 关键点绘制时缩放尺度
POSE_SCALES_LIST = [
    POSE_BODY_25_SCALES
]
# 关键点着色方案
POSE_COLORS_LIST = [
    POSE_BODY_25_COLORS
]


# 错误信息提示
ERROR_MODEL = "模型错误！请使用正确的OpenPose模型！"
ERROR_MODEL_INDEX = "模型下标越界！请完善或添加模型对应的参数！"


def _get_model_param(model: OpModel, params: list):
    """
    获取params中对应于model的参数
    :param model:
    :param params:
    :return:
    """
    if isinstance(model, OpModel):
        if model.value < len(params):
            return params[model.value]
        else:
            raise IndexError(ERROR_MODEL_INDEX)
    else:
        raise ValueError(ERROR_MODEL)


def get_pose_part_number(model):
    """
    获得模型对应的身体关键点数目
    :param model:
    :return:
    """
    return _get_model_param(model, POSE_PART_NUMBER_LIST)


def get_pose_part_mapping(model):
    """
    获取模型对应的关键点映射
    :param model:
    :return:
    """
    return _get_model_param(model, POSE_PART_MAPPING_LIST)


def get_pose_part_pairs(model):
    """
    获取模型对应的关键点对
    :param model:
    :return:
    """
    return _get_model_param(model, POSE_PART_PAIRS_LIST)


def get_pose_scales(model):
    """
    获取model对应的绘制时的缩放尺度
    :param model:
    :return:
    """
    return _get_model_param(model, POSE_SCALES_LIST)


def get_pose_colors(model):
    """
    获取model对应的绘制时的着色方案
    :param model:
    :return:
    """
    return _get_model_param(model, POSE_COLORS_LIST)


if __name__ == "__main__":
    m = OpModel.BODY_25
    print(get_pose_part_number(m))
    print(get_pose_part_mapping(m))
    print(get_pose_part_pairs(m))
    print(get_pose_scales(m))
    print(get_pose_colors(m))
