# -*- coding: utf-8 -*-

import random
import os.path as osp
import json
import cv2

import torchvision.transforms as trans

from model.resnet import ResNet
from data.dataset.dataloader import dataset_reader, get_snippet_indices, snippets_list_concat
from myutils.func import *


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


def main(dataset_folder, epoch_train_num=30):
    model = ResNet()

    data_list = dataset_reader(dataset_folder)
    # 只使用前200人的数据，且以cam_a为probe，cam_b为gallery
    probe_set, gallery_set = data_list[0][:200], data_list[1][:200]

    # mean, std = channel_mean_std([probe_set[:], gallery_set[:]])
    # exit(0)
    mean, std = [0.4348, 0.4756, 0.3638], [0.1634, 0.1705, 0.1577]

    transforms = trans.Compose([
        trans.ToTensor(),
        trans.Normalize(mean=mean, std=std)
    ])

    # 在不同的分割上进行同样次数的训练
    split_num = 10
    split_list = get_splits('./prid2011_splits.json')
    for i in range(split_num):
        cur_split = split_list[i]
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

        print(list_shape(train_probe_snippets), list_shape(train_gallery_snippets),
              list_shape(test_probe_snippets), list_shape(test_gallery_snippets))
        print(list_shape(train_probe_labels), list_shape(train_gallery_labels),
              list_shape(test_probe_labels), list_shape(test_gallery_labels))
        print(list_shape(trp_start_indices), list_shape(trg_start_indices),
              list_shape(tep_start_indices), list_shape(teg_start_indices))
        # print(train_probe_labels)
        # (3520, 8, 128, 64, 3) (2637, 8, 128, 64, 3) (3865, 8, 128, 64, 3) (2532, 8, 128, 64, 3)
        # (3520,)(2637, )(3865, )(2532, )
        # (101,)(101, )(101, )(101, )

        # 数据预处理

        # shuffle, not now, todo

        # break
        # 训练
        for snippet_index, snippet in enumerate(train_probe_snippets):
            snippet_anchor = snippet[:]     # 拷贝
            person_label = train_probe_labels[snippet_index]

            # 在gallery中找一个正样本
            positive_start_index_index = 0
            for t in range(len(trg_start_indices)-1):
                start_index_t, start_index_tn = trg_start_indices[i], trg_start_indices[i+1]
                if start_index_t <= snippet_index < start_index_tn:
                    positive_start_index_index = t
                    break
            positive_snippet_index = random.randrange(trg_start_indices[positive_start_index_index],
                                                      trg_start_indices[positive_start_index_index+1])
            snippet_positive = train_gallery_snippets[positive_snippet_index]

            # 在gallery中找一个负样本
            negative_start_index_index = rand_range(0, len(trg_start_indices)-1, positive_start_index_index)
            negative_snippet_index = random.randrange(trg_start_indices[negative_start_index_index],
                                                      trg_start_indices[negative_start_index_index+1])
            snippet_negative = train_gallery_snippets[negative_snippet_index]

            print("anchor行人索引：{}\tpositive行人索引：{}\tnegative行人索引：{}".format(person_label,
                                                                           train_gallery_labels[positive_snippet_index],
                                                                           train_gallery_labels[negative_snippet_index]))

            # show examples
            # draw_4d_list(snippet_anchor, "anchor")
            # draw_4d_list(snippet_positive, "positive")
            # draw_4d_list(snippet_negative, "negative")
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            print(type(snippet_anchor), type(snippet_anchor[0]))

            snippet_anchor = transforms(np.asarray(snippet_anchor))
            snippet_positive = transforms(np.asarray(snippet_positive))
            snippet_negative = transforms(np.asarray(snippet_negative))

            anchor_out = model(snippet_anchor)
            positive_out = model(snippet_positive)
            negative_out = model(snippet_negative)





            break
        break


if __name__ == '__main__':
    main("D:/datasets/dst/prid_2011/src")
    # splits = get_splits('./prid2011_splits.json', person_num=200)
    # print(splits)
    # for s in splits:
    #     print(s["split_id"], len(s["train_ids"]), len(s["test_ids"]))
    # print("over!")
