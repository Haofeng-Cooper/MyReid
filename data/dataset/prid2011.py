# -*- coding: utf-8 -*-
# 生成prid2011的pose图数据

from data.dataset.datasetgenerate import MyDataSet


if __name__ == "__main__":
    dataset = MyDataSet("D:/test/dataset", "../openpose/bin/OpenPoseDemo.exe")
    dataset.fill_folder_list()
    dataset.process(tmp_folder="D:/test/tmp", dst_folder="D:/test/result", image_shape=(128, 64), image_format=".png")
