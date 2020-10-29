# -*- coding: utf-8 -*-

import random
import os.path as osp
import json
import cv2

import torch
from torch import optim
import torchvision.transforms as trans

from model.resnet import ResNet
from data.dataset.dataloader import dataset_reader, get_snippet_indices, snippets_list_concat
from myutils.func import *
from myutils.fileutils import write_json_file, read_json_file

from data.dataset.dataloader import dataset_reader2, train_data_loader


def get_splits(filepath, person_num=200, train_rate=0.5, repeat_num=10):
    """
    获取train/test分割,
    其中：
        person_num=200是指prid2011数据集只使用前200人的一共400个视频序列，是为了和前人的工作保持一致。
        train_rate=0.5指一半用于训练，一半用于测试，也是为了和前人的工作保持一致。
        repeat_num=10指重复10次，也即随机进行10次train/test分割（最后的结果取均值），同样是为了和前人的工作保持一致。
    """
    train_num = positive_round(person_num * train_rate)

    if osp.isfile(filepath):
        with open(filepath, 'r') as f:
            json_data = json.load(fp=f)
            p_num, t_num, r_num = json_data["person_num"], json_data["train_num"], len(json_data["splits"])
            if p_num == person_num and t_num == train_num and r_num == repeat_num:
                print("使用分割文件：", filepath)
                return json_data['splits']

    print("重新生成分割文件：", filepath)
    splits = []
    person_ids = list(range(person_num))
    for i in range(repeat_num):
        train_ids = random.sample(person_ids, train_num)
        train_ids.sort()
        test_ids = list(set(person_ids).difference(set(train_ids)))

        d = {'split_id': i, "train_ids": train_ids, "test_ids": test_ids}
        splits.append(d)
    write_json_file({"person_num": person_num, "train_num": train_num,  "splits": splits}, filepath)
    return splits


def get_splits2(save_file, person_ids: list, train_rate=0.5, repeat_num=10, shuffle=True, force_re=False):
    """
    对行人索引列表进行train/test分割，并将结果保存到指定文件中。
    如果文件已存在，则优先从文件中读取数据。

    :param save_file: 保存分割信息的json文件
    :param person_ids: 要进行train/test分割的行人索引列表
    :param train_rate: 训练数据占比，剩余为测试数据
    :param repeat_num: 分割重复次数
    :param shuffle: 是否对训练和测试的行人索引打乱顺序
    :param force_re: 是否强制重新生成分割
    """
    person_num = len(person_ids)
    train_num = positive_round(person_num * train_rate)

    if not force_re and osp.isfile(save_file):
        json_data = read_json_file(save_file)
        p_num, t_num, r_num = json_data["person_num"], json_data["train_num"], len(json_data["splits"])
        if p_num == person_num and t_num == train_num and r_num == repeat_num:
            print("使用分割文件：", osp.abspath(save_file))
            return json_data['splits']

    print("重新生成分割文件：", save_file)
    splits = []
    for i in range(repeat_num):
        train_ids = random.sample(person_ids, train_num)
        test_ids = list(set(person_ids).difference(set(train_ids)))

        if shuffle:
            random.shuffle(train_ids)
            random.shuffle(test_ids)
        cur_split = {'split_id': i, "train_ids": train_ids, "test_ids": test_ids}
        splits.append(cur_split)
    write_json_file({"person_num": person_num, "train_num": train_num,  "splits": splits}, save_file)
    return splits


def sample_dataset(seq_list, snippet_len=8, stride=3):
    """
    对seq_list的每一个seq分段，每个seq都是一个行人在某个相机下的所有图像。
    返回所有人的所有snippet
    """
    result = []
    for seq in seq_list:
        result.append(get_snippet_indices(seq=seq, snippet_len=snippet_len, stride=stride, only_index=False))
    return result


def element_index(e_index, start_index_list):
    """获取元素对应的下标"""
    for i in range(len(start_index_list)-1):
        start_index, end_index = start_index_list[i], start_index_list[i+1]
        if start_index <= e_index < end_index:
            return i
    return -1


def main2(dataset_folder, data_npz_file, epoch_train=30):
    image_array, seq_range_array, cam_offset_array = dataset_reader2(dataset_folder, data_npz_file, force_re=False)
    # 取第一个相机下的所有行人索引作为整个数据集的所有行人索引
    person_ids = list(seq_range_array[:, 0][cam_offset_array[0]:cam_offset_array[1]])
    # 取第一个相机为probe，第二个相机为gallery，则第二个相机下第一个人的起始帧索引为probe和gallery的分割索引
    probe_gallery_split_index = seq_range_array[cam_offset_array[1]][1]
    probe_set = image_array[:probe_gallery_split_index]
    gallery_set = image_array[probe_gallery_split_index:]

    # 计算整个数据集上数据的均值和标准差
    # mean_value, std_value = [0.0 for _ in range(3)], [0.0 for _ in range(3)]
    # for i in range(3):
    #     channel_data = image_array[:, :, :, i] / 255
    #     mean_value[i] = round(np.mean(channel_data), 4)
    #     std_value[i] = round(np.std(channel_data), 4)
    mean_value, std_value = [0.4348, 0.4756, 0.3638], [0.1634, 0.1705, 0.1577]

    splits = get_splits2("./prid2011_splits.json", person_ids, shuffle=True, force_re=False)
    batch_size = 8
    transforms = trans.Compose([
        trans.ToTensor(),           # 如果是numpy-ndarray且dtype=unint8会自动除以255，且维度由(h, w, c)变成(c, h, w)
        trans.Normalize(mean=mean_value, std=std_value)
    ])

    device = (torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"))
    net = ResNet(out_features=1000, final_pool="avg")
    net = net.cuda(device=device)

    triplet_loss = torch.nn.TripletMarginLoss(margin=1.0, p=2)
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for split_id, split in enumerate(splits):
        train_ids, test_ids = split["train_ids"], split["test_ids"]
        train_data = train_data_loader(train_ids, image_array, seq_range_array, cam_offset_array, batch_size=batch_size)

        for epoch_id in range(epoch_train):
            for batch_id, batch_data in enumerate(train_data):
                batch_loss = 0.0
                for anc_sample, pos_sample, neg_sample in batch_data:
                    optimizer.zero_grad()

                    # print(anc_sample.shape, pos_sample.shape, neg_sample.shape)
                    # draw_4d_list(anc_sample, "anchor", max_cols=10)
                    # draw_4d_list(pos_sample, "positive", max_cols=10)
                    # draw_4d_list(neg_sample, "negative", max_cols=10)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()
                    # exit(1)

                    # ndarray转变为tensor和数据预处理
                    input_anchor = ndarray4d_2_tensor(anc_sample, transforms, grad=True)
                    input_positive = ndarray4d_2_tensor(pos_sample, transforms, grad=True)
                    input_negative = ndarray4d_2_tensor(neg_sample, transforms, grad=True)

                    # 转移到cuda上
                    input_anchor = input_anchor.to(device)
                    input_positive = input_positive.to(device)
                    input_negative = input_negative.to(device)
                    # print("input_anchor", input_anchor.shape)

                    # 送入网络获取输出
                    output_anchor = net(input_anchor)
                    output_positive = net(input_positive)
                    output_negative = net(input_negative)
                    # print("output_anchor", output_anchor.shape)

                    # 计算损失
                    loss = triplet_loss(output_anchor, output_positive, output_negative)
                    loss.backward()  # 误差反向传播，计算梯度
                    optimizer.step()  # 更新权重

                    batch_loss += loss.item()
                batch_loss = batch_loss / batch_size
                print("split: %d \t epoch: %d \t batch: %d \t batch_avg_loss:%f" %
                      (split_id, epoch_id, batch_id, batch_loss))


def main(dataset_folder, epoch_train_num=30):
    device = (torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"))

    model = ResNet(out_features=1000, final_pool="avg")
    model.to(device)

    data_list = dataset_reader(dataset_folder)
    # 只使用前200人的数据，且以cam_a为probe，cam_b为gallery
    probe_set, gallery_set = data_list[0][:200], data_list[1][:200]

    # mean, std = channel_mean_std([probe_set[:], gallery_set[:]])
    # exit(0)
    mean, std = [0.4348, 0.4756, 0.3638], [0.1634, 0.1705, 0.1577]

    transforms = trans.Compose([
        trans.ToTensor(),           # 如果是numpy-ndarray且dtype=unint8会自动除以255，且维度由(h, w, c)变成(c, h, w)
        trans.Normalize(mean=mean, std=std)
    ])

    triplet_loss = torch.nn.TripletMarginLoss(margin=1.0, p=2)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # 在不同的分割上进行同样次数的训练
    split_num = 10
    split_list = get_splits('./prid2011_splits.json')
    for split_id in range(split_num):
        cur_split = split_list[split_id]
        cur_train_ids, cur_test_ids = cur_split["train_ids"], cur_split["test_ids"]
        # 获取当前分割对应的训练和测试数据
        cur_train_probe_list = shuffle_list(probe_set, cur_train_ids)
        cur_test_probe_list = shuffle_list(probe_set, cur_test_ids)
        cur_train_gallery_list = shuffle_list(gallery_set, cur_train_ids)
        cur_test_gallery_list = shuffle_list(gallery_set, cur_test_ids)

        # print(list_shape(cur_train_probe_list), list_shape(cur_test_probe_list),
        #       list_shape(cur_train_gallery_list), list_shape(cur_test_gallery_list))

        # 训练和测试数据进行片段划分
        train_probe_snippets_list = sample_dataset(cur_train_probe_list)
        train_gallery_snippets_list = sample_dataset(cur_train_gallery_list)
        test_probe_snippets_list = sample_dataset(cur_test_probe_list)
        test_gallery_snippets_list = sample_dataset(cur_test_gallery_list)
        # 每一个snippets_list，[i]=e，e是train/test-probe/gallery下第i个行人的所有视频片段list
        # (person_num * snippets_num * H * W * 3)，其中snippets_num是不相同的

        # 将所有人的所有片段拼接到一块，获取每个片段对应的真实行人标签
        train_probe_snippets, train_probe_labels, trp_start_indices = snippets_list_concat(train_probe_snippets_list, cur_train_ids)
        train_gallery_snippets, train_gallery_labels, trg_start_indices = snippets_list_concat(train_gallery_snippets_list, cur_train_ids)
        test_probe_snippets, test_probe_labels, tep_start_indices = snippets_list_concat(test_probe_snippets_list, cur_test_ids)
        test_gallery_snippets, test_gallery_labels, teg_start_indices = snippets_list_concat(test_gallery_snippets_list, cur_test_ids)

        # print(list_shape(train_probe_snippets), list_shape(train_gallery_snippets),
        #       list_shape(test_probe_snippets), list_shape(test_gallery_snippets))
        # print(list_shape(train_probe_labels), list_shape(train_gallery_labels),
        #       list_shape(test_probe_labels), list_shape(test_gallery_labels))
        # print(list_shape(trp_start_indices), list_shape(trg_start_indices),
        #       list_shape(tep_start_indices), list_shape(teg_start_indices))
        # (3520, 8, 128, 64, 3) (2637, 8, 128, 64, 3) (3865, 8, 128, 64, 3) (2532, 8, 128, 64, 3)
        # (3520,)(2637, )(3865, )(2532, )
        # (101,)(101, )(101, )(101, )

        # shuffle, not now, todo

        for epoch in range(epoch_train_num):
            running_loss = 0.0
            running_loss_avg_num = 100

            # 训练
            snippet_index = 0
            for snippet_index, snippet in enumerate(train_probe_snippets):
                optimizer.zero_grad()

                snippet_anchor = snippet[:]  # 拷贝
                person_label = train_probe_labels[snippet_index]

                # 在gallery中找一个正样本
                positive_start_index_index = 0
                for t in range(len(trg_start_indices) - 1):
                    start_index_t, start_index_tn = trg_start_indices[split_id], trg_start_indices[split_id + 1]
                    if start_index_t <= snippet_index < start_index_tn:
                        positive_start_index_index = t
                        break
                positive_snippet_index = random.randrange(trg_start_indices[positive_start_index_index],
                                                          trg_start_indices[positive_start_index_index + 1])
                snippet_positive = train_gallery_snippets[positive_snippet_index]

                # 在gallery中找一个负样本
                negative_start_index_index = rand_range(0, len(trg_start_indices) - 1, positive_start_index_index)
                negative_snippet_index = random.randrange(trg_start_indices[negative_start_index_index],
                                                          trg_start_indices[negative_start_index_index + 1])
                snippet_negative = train_gallery_snippets[negative_snippet_index]

                print("anchor行人索引：{}\tpositive行人索引：{}\tnegative行人索引：{}".format(
                    person_label, train_gallery_labels[positive_snippet_index],
                    train_gallery_labels[negative_snippet_index]))

                # show examples
                # draw_4d_list(snippet_anchor, "anchor")
                # draw_4d_list(snippet_positive, "positive")
                # draw_4d_list(snippet_negative, "negative")
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

                # print(type(snippet_anchor), type(snippet_anchor[0]))
                # <class 'list'> <class 'numpy.ndarray'>

                snippet_anchor = ndarray_list_2_tensor(snippet_anchor, transforms)
                snippet_positive = ndarray_list_2_tensor(snippet_positive, transforms)
                snippet_negative = ndarray_list_2_tensor(snippet_negative, transforms)

                snippet_anchor = snippet_anchor.to(device)
                snippet_positive = snippet_positive.to(device)
                snippet_negative = snippet_negative.to(device)

                anchor_out = model(snippet_anchor)
                positive_out = model(snippet_positive)
                negative_out = model(snippet_negative)
                # print(anchor_out)
                # exit(0)
                # print(anchor_out.shape, positive_out.shape, negative_out.shape)
                # out size: (out_features)

                anchor_out = torch.unsqueeze(anchor_out, dim=0)
                positive_out = torch.unsqueeze(positive_out, dim=0)
                negative_out = torch.unsqueeze(negative_out, dim=0)
                # print(anchor_out.shape, positive_out.shape, negative_out.shape)
                # out size: (1 x out_features)
                exit(0)

                loss = triplet_loss(anchor_out, positive_out, negative_out)
                loss.backward()  # 误差反向传播，计算梯度
                optimizer.step()  # 更新权重

                running_loss += loss.item()
                if (snippet_index+1) % running_loss_avg_num == 0:
                    print("split: %d\t epoch: %d\t snippet: %d\t avg_loss: %.3f" %
                          (split_id + 1, epoch + 1, snippet_index + 1, running_loss / running_loss_avg_num))
                    running_loss = 0.0
            print("split: %d\t epoch: %d\t snippet: %d\t avg_loss: %.3f" %
                  (split_id + 1, epoch + 1, snippet_index + 1, running_loss / running_loss_avg_num))

            # 计算该epoch后的测试mAP


if __name__ == '__main__':
    # main("/home/haofeng/Desktop/datasets/dst/prid_2011/src")
    # splits = get_splits('./prid2011_splits.json', person_num=200)
    # print(splits)
    # for s in splits:
    #     print(s["split_id"], len(s["train_ids"]), len(s["test_ids"]))
    # print("over!")

    main2(dataset_folder="/home/haofeng/Desktop/datasets/dst/prid_2011/src",
          data_npz_file="/home/haofeng/Desktop/datasets/dst/prid_2011/src_array.npz",
          epoch_train=40)
