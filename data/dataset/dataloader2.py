import os.path as osp
from glob import glob
import cv2
import numpy as np
import random
from math import ceil

from myutils.func import print_process, positive_round
from myutils.fileutils import read_json_file, write_json_file
from myutils.listutils import list_choice, list_relist


def __images_reader(images_folder, save_file, image_suffix, force_re=False):
    if not force_re and osp.isfile(save_file):
        # 数据已经读取并存储过，且不需要重新读取
        print("使用数据文件：{}".format(save_file))
        # todo
        data = np.load(save_file)
        image_array, seq_range = data["arr_0"], data["arr_1"]
        return image_array, seq_range

    if not osp.isdir(images_folder):
        raise ValueError("目标文件夹不存在！{}".format(images_folder))

    files_list = list(sorted(glob(osp.join(images_folder, image_suffix))))
    files_num = len(files_list)

    image_array = []
    seq_range = []     # 记录每个相机下每个行人图像的序列在 image_array 中的起始和终止下标，长度为所有相机下的视频数总和

    last_cam_id, last_person_id, last_image_id = 0, 0, 0
    last_seq_start_index = 0
    print("开始读取目录{}下的图像文件...".format(images_folder))
    for index, filename in enumerate(files_list):
        filename_base = osp.basename(filename)
        print_process(cur=index+1, total=files_num, extra_info=filename_base)
        name_list = filename_base.split('.')[0].split('_')
        cam_id, person_id, image_id = int(name_list[0]), int(name_list[1]), int(name_list[2])

        if image_id < last_image_id:
            # 表明上一个行人的图像序列结束了
            seq_range.append([last_cam_id, last_person_id, last_seq_start_index, index])
            last_seq_start_index = index

        if person_id < last_person_id:
            # 表明上一个相机下的所有图像序列都结束了
            last_cam_id = cam_id

        last_person_id, last_image_id = person_id, image_id
        image_array.append(cv2.imread(filename))
    seq_range.append([last_cam_id, last_person_id, last_seq_start_index, files_num])

    max_cam, max_person = seq_range[-1][0]+1, seq_range[-1][1]+1
    t_list = seq_range[:]
    seq_range = np.zeros(shape=(max_cam, max_person, 2), dtype=np.int)
    for t_info in t_list:
        seq_range[t_info[0]][t_info[1]] = t_info[2], t_info[3]
    # 打印相关信息
    # print(seq_range.shape)
    # print(seq_range)

    # 将数据信息转变成numpy.ndarray数组存储
    image_array = np.asarray(image_array)
    # seq_range = np.asarray(seq_range)
    np.savez(save_file, image_array, seq_range)
    print("存储数据到文件：{}".format(save_file))

    return image_array, seq_range


def dataset_reader(dataset_folder, data_mode="src", image_suffix="*.png", force_re=False):
    """
    读取目标文件夹下的指定数据类型的数据，返回 numpy.ndarray 类型的图像数据，以及每个相机下每个人的在最终数据中的起始和结束索引。

    要求图像文件命名格式为 "相机索引_行人索引_图像索引.png"，且各索引都从0开始。

    同时将读到的图像数据写入到文件中，在下次读取时不用循环读取每个图像文件，只需要读取该文件即可。

    优先读取包含所有图像数据的文件而不是从原始图像文件中读取数据。

    :param dataset_folder: 存储图像文件的文件夹的父文件夹
    :param data_mode: 读取哪种类型的图像数据。
                      src: 原始图像帧；pose：姿态/骨架图；src_pose：原始图像帧和姿态/骨架图，且在通道层进行拼接。
                      要求原始图像帧与姿态/骨架图一一对应
    :param image_suffix: 图像文件的后缀名，如 *.png 表示读取所有.png文件
    :param force_re: 为True则强制依次从图像文件中重新读取数据
    """

    if not osp.isdir(dataset_folder):
        raise ValueError("目标文件夹不存在！{}".format(dataset_folder))

    mode_list = ["src", "pose", "src_pose"]
    if data_mode not in mode_list:
        raise ValueError("'{}'是不支持的数据读取类型！支持的数据读取类型为{}".format(data_mode, mode_list))

    src_file = osp.join(dataset_folder, "src_ndarray.npz")
    pose_file = osp.join(dataset_folder, "pose_ndarray.npz")

    src_folder = osp.join(dataset_folder, "src")
    pose_folder = osp.join(dataset_folder, "pose")

    # data_array, seq_range = None, None
    if data_mode == "src":
        data_array, seq_range = __images_reader(src_folder, save_file=src_file,
                                                image_suffix=image_suffix, force_re=force_re)
    elif data_mode == "pose":
        data_array, seq_range = __images_reader(pose_folder, save_file=pose_file,
                                                image_suffix=image_suffix, force_re=force_re)
    else:
        src_array, seq_range = __images_reader(src_folder, save_file=src_file,
                                               image_suffix=image_suffix, force_re=force_re)
        pose_array, seq_range = __images_reader(pose_folder, save_file=pose_file,
                                                image_suffix=image_suffix, force_re=force_re)
        data_array = np.concatenate([src_array, pose_array], axis=3)
    # print(data_array.shape, seq_range.shape)
    return data_array, seq_range


def get_split(split_id, split_file, person_num, train_rate=0.5, repeat_num=10, force_re=False):
    """
    从行人ID分割文件中读取目标分割索引的数据（split_id，train_ids，test_ids）

    如果文件不存在或者强制要求重新生成，重新生成分割文件。

    :param split_id: 要使用的分割索引
    :param split_file: 保存分割信息的json文件
    :param person_num: 数据中的行人数目（每个摄像机下应该是相同的）
    :param train_rate: 用于训练的行人数占比，剩余为测试数据
    :param repeat_num: 分割的重复次数
    :param force_re: 是否强制重新生成分割
    """
    person_ids = list(range(person_num))
    train_num = positive_round(person_num * train_rate)

    if not force_re and osp.isfile(split_file):
        json_data = read_json_file(split_file)
        p_num, t_num, r_num = json_data["person_num"], json_data["train_num"], len(json_data["splits"])
        if p_num == person_num and t_num == train_num and r_num == repeat_num:
            print("使用分割文件：", osp.abspath(split_file))
            split_data = json_data["splits"][split_id]
            return split_data

    splits = []
    for i in range(repeat_num):
        train_ids = random.sample(person_ids, train_num)
        test_ids = list(set(person_ids).difference(set(train_ids)))

        cur_split = {'split_id': i, "train_ids": sorted(train_ids), "test_ids": sorted(test_ids)}
        splits.append(cur_split)
    write_json_file({"person_num": person_num, "train_num": train_num,  "splits": splits}, split_file)
    print("重新生成分割文件：", osp.abspath(split_file))
    return splits[split_id]


def seq2snippets(seq, snippet_len=8, snippet_stride=4):
    """
    帧序列变为多个帧片段
    返回帧片段列表

    :param seq: 帧序列
    :param snippet_len: 片段的帧长度
    :param snippet_stride: 片段划分的步长
    """
    snippets = []
    seq_len = len(seq)
    if seq_len <= snippet_len:
        snippet_i = []
        for i in range(seq_len):
            snippet_i.append(seq[i])
        for i in range(seq_len, snippet_len):
            snippet_i.append(seq[-1])
        snippets.append(snippet_i)
    else:
        snippet_num = (seq_len-snippet_len) // snippet_stride + 1
        for snippet_id in range(snippet_num):
            snippet_i = []
            start_fid = snippet_id * snippet_stride
            stop_fid = start_fid + snippet_len
            for frame_id in range(start_fid, stop_fid):
                snippet_i.append(seq[frame_id])
            snippets.append(snippet_i)
    return snippets


def train_data_loader(train_ids, pg_array, seq_range, batch_size=8, shuffle=True,
                      snippet_len=8, snippet_stride=4):
    """
    从pg_array中加载用于训练的数据，进行片段划分，然后分batch，返回每个batch的 anchor，positive，negative样本数据

    默认probe数据的cam_id=0，gallery数据的cam_id=1

    :param train_ids 用于训练数据的行人索引
    :param pg_array probe和gallery的所有图像数据
    :param seq_range 各相机下的各个行人的视频序列的起始和终止索引 [[cam_id, pid, start_index, end_index]]
    :param batch_size 批量大小
    :param shuffle 是否打乱顺序
    :param snippet_len 片段的帧长度
    :param snippet_stride 划分片段时的步长
    """
    p_cam, g_cam = 0, 1

    # 获取训练行人的所有片段分割
    p_snippets, g_snippets = [], []
    for person_id in train_ids:
        p_seq_info = seq_range[p_cam][person_id]
        g_seq_info = seq_range[g_cam][person_id]

        p_seq = pg_array[p_seq_info[0]: p_seq_info[1]]
        g_seq = pg_array[g_seq_info[0]: g_seq_info[1]]

        p_s = seq2snippets(seq=p_seq, snippet_len=snippet_len, snippet_stride=snippet_stride)
        g_s = seq2snippets(seq=g_seq, snippet_len=snippet_len, snippet_stride=snippet_stride)

        p_snippets.append(p_s)
        g_snippets.append(g_s)

    # 为每一个probe片段选择一个正样本和负样本，组合在一起
    all_apn, all_apn_labels = [], []
    index_list = list(range(len(train_ids)))
    for index, probe_snippet in enumerate(p_snippets):
        for anchor in probe_snippet:
            ap_pid = train_ids[index]
            positive = random.choice(g_snippets[index])

            neg_index = list_choice(data_list=index_list, exclude=index)
            n_pid = train_ids[neg_index]
            negative = random.choice(g_snippets[neg_index])

            all_apn.append([anchor, positive, negative])
            all_apn_labels.append([ap_pid, ap_pid, n_pid])       # 对应的真实行人索引

    # 顺序打乱
    apn_num = len(all_apn)
    if shuffle:
        shuffled_index_list = list(range(apn_num))
        random.shuffle(list(range(len(all_apn))))

        all_apn = list_relist(data_list=all_apn, index_list=shuffled_index_list)
        all_apn_labels = list_relist(data_list=all_apn_labels, index_list=shuffled_index_list)

    # batch划分
    batch_num = ceil(apn_num / batch_size)
    # print("共有{}个片段，划分成{}个batch".format(apn_num, batch_num))
    result = []
    index = 0
    for i in range(batch_num):
        batch_data = []
        for j in range(batch_size):
            real_index = index % apn_num
            batch_data.append([all_apn[real_index], all_apn_labels[real_index]])
            index += 1
        result.append(batch_data)
    return result


def test_data_loader(test_ids, pg_array, seq_range, snippet_len=8, snippet_stride=4):
    """
    从pg_array中加载测试数据，返回probe和gallery的下所有测试行人的所有片段
    视频序列分段不分批
    """
    p_cam, g_cam = 0, 1
    p_snippets, g_snippets = [], []
    for pid in test_ids:
        p_seq_info = seq_range[p_cam][pid]
        g_seq_info = seq_range[g_cam][pid]

        p_seq = pg_array[p_seq_info[0]: p_seq_info[1]]
        g_seq = pg_array[g_seq_info[0]: g_seq_info[1]]

        p_s = seq2snippets(seq=p_seq, snippet_len=snippet_len, snippet_stride=snippet_stride)
        g_s = seq2snippets(seq=g_seq, snippet_len=snippet_len, snippet_stride=snippet_stride)

        p_snippets.append(p_s)
        g_snippets.append(g_s)
    return p_snippets, g_snippets


if __name__ == '__main__':
    # dataset_reader("/home/haofeng/Desktop/datasets/dst/prid_2011", data_mode="src", force_re=True)
    dataset_reader("/home/haofeng/Desktop/datasets/dst/prid_2011", data_mode="src_pose", force_re=True)

    print(get_split(0, "/home/haofeng/Desktop/datasets/dst/prid_2011/prid2011_splits.json", 200, force_re=True))
