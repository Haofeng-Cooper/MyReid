# -*- coding: utf-8 -*-

# 在渲染openpose关键点图像时用到的函数，参考自openpose源代码（c++版）


def get_keypoints_rectangle(keypoints, thresholdRectangle):
    """
    获取关键点所在矩形框
    :param keypoints: 浮点数组，长度为：关键点数目n x 3（x，y, score）
    :param thresholdRectangle: 阈值
    :return:
    """
    minX, minY = 10000, 10000
    maxX, maxY = -1, -1
    for i in range(0, len(keypoints), 3):
        x = keypoints[i]
        y = keypoints[i+1]
        score = keypoints[i+2]

        if score > thresholdRectangle:
            if x < minX: minX = x
            if x > maxX: maxX = x
            if y < minY: minY = y
            if y > maxY: maxY = y
    if maxX >= minX and maxY > minY:
        return (minX, minY, maxX - minX, maxY - minY)
    else:
        return (0,0,0,0)

def rectangle_width(rectangle):
    return rectangle[2]

def rectangle_height(rectangle):
    return rectangle[3]

def area(rectangle):
    return rectangle[2] * rectangle[3]

def positive_int_round(x):
    '''
    四舍五入
    :param x: 浮点数x
    :return: 四舍五入后的整数
    '''
    return int(x + 0.5)

