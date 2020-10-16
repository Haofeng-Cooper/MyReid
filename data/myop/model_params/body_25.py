# -*- coding: utf-8 -*-

# body25模型参数


# 关键点数目
POSE_BODY_25_PART_NUMBER = 25
# 索引和关键点映射
POSE_BODY_25_PART_MAPPING = {0: 'Nose', 1: 'Neck', 2: 'RShoulder', 3: 'RElbow', 4: 'RWrist', 5: 'LShoulder', 6: 'LElbow', 7: 'LWrist', 8: 'MidHip', 9: 'RHip', 10: 'RKnee', 11: 'RAnkle', 12: 'LHip', 13: 'LKnee', 14: 'LAnkle', 15: 'REye', 16: 'LEye', 17: 'REar', 18: 'LEar', 19: 'LBigToe', 20: 'LSmallToe', 21: 'LHeel', 22: 'RBigToe', 23: 'RSmallToe', 24: 'RHeel', 25: 'Background'}
# 关键点对（关键点之间可连线）
POSE_BODY_25_PART_PAIRS = [1,8,   1,2,   1,5,   2,3,   3,4,   5,6,   6,7,   8,9,   9,10,  10,11, 8,12,  12,13, 13,14,  1,0,   0,15, 15,17,  0,16, 16,18,   14,19,19,20,14,21, 11,22,22,23,11,24]


# 绘制时缩放尺度
POSE_BODY_25_SCALES = [1.]
# 绘制时各关键点RGB色值
POSE_BODY_25_COLORS = [
    255., 0., 85.,
    255., 0., 0.,
    255., 85., 0.,
    255., 170., 0.,
    255., 255., 0.,
    170., 255., 0.,
    85., 255., 0.,
    0., 255., 0.,
    255., 0., 0.,
    0., 255., 85.,
    0., 255., 170.,
    0., 255., 255.,
    0., 170., 255.,
    0., 85., 255.,
    0., 0., 255.,
    255., 0., 170.,
    170., 0., 255.,
    255., 0., 255.,
    85., 0., 255.,
    0., 0., 255.,
    0., 0., 255.,
    0., 0., 255.,
    0., 255., 255.,
    0., 255., 255.,
    0., 255., 255.
]
