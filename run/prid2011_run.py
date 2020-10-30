# -*- coding: utf-8 -*-

import random
import os.path as osp
import json
import cv2

import torch
from torch import optim
import torchvision.transforms as trans

from model.resnet import ResNet
from myutils.func import *
from myutils.fileutils import write_json_file, read_json_file
from myutils.listutils import list_sort
from myutils.calc import AverageCalc
from data.dataset.dataloader import dataset_reader2, train_data_loader2, test_data_loader


def get_splits(save_file, person_ids: list, train_rate=0.5, repeat_num=10, shuffle=True, force_re=False):
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


@torch.no_grad()
def check_top_k(test_data, probe_ids, gallery_ids, transforms, net_model, device, top_k=(1, 5, 10, 20)):
    # person_num = len(test_ids)
    probe_seqs, gallery_seqs = test_data[0], test_data[1]
    probe_num, gallery_num = len(probe_seqs), len(gallery_seqs)
    # 先对每一个seq生成对应的特征向量
    probe_fvs, gallery_fvs = [], []
    for i in range(probe_num):
        p_seq = ndarrays_2_tensor(probe_seqs[i], transforms, grad=False)
        p_seq = p_seq.to(device)
        p_fv = net_model(p_seq)
        probe_fvs.append(torch.unsqueeze(p_fv, dim=0))
    for i in range(gallery_num):
        g_seq = ndarrays_2_tensor(gallery_seqs[i], transforms, grad=False)
        g_seq = g_seq.to(device)
        g_fv = net_model(g_seq)
        gallery_fvs.append(torch.unsqueeze(g_fv, dim=0))

    # 计算特征向量的距离或相似度
    dist_matrix = np.zeros(shape=(probe_num, gallery_num), dtype=np.float)
    for i in range(probe_num):
        for j in range(gallery_num):
            dist_matrix[i][j] = torch.pairwise_distance(probe_fvs[i], gallery_fvs[j]).item()

    # 获得按距离由小到大或相似度由大到小的gallery-label二维数组，一维为对应的probe下标
    pid_matrix = []
    for i in range(probe_num):
        pid_matrix.append(list_sort(gallery_ids, dist_matrix[i], "ASC"))

    # 计算每个probe的top-k
    top_k_matrix = np.zeros(shape=(probe_num, len(top_k)), dtype=np.int)
    for i in range(probe_num):
        for j in range(len(top_k)):
            k = top_k[j]
            probe_id = probe_ids[i]
            first_k = list(pid_matrix[i][0:k])
            if probe_id in first_k:
                top_k_matrix[i][j] = 1

    # 计算整个probe上的平均top-k
    avg_top_k = np.mean(top_k_matrix, axis=0)     # 按列求均值
    return avg_top_k


def is_better_top_k(old_top_k, new_top_k):
    for i in range(len(old_top_k)):
        if new_top_k[i] > old_top_k[i]:
            return True
    return False


def get_model_params_file_name(k, top_k):
    filename = "model_1_params"
    for i in range(len(k)):
        filename += ("_top%d_%.3f" % (k[i], top_k[i]))
    filename += ".pth"
    return filename


def save_model_params(k, top_k, model_folder, model):
    filename = get_model_params_file_name(k, top_k)
    filepath = osp.join(model_folder, filename)
    torch.save(model.state_dict(), filepath)
    print("保存模型参数到：" + filepath)


def main(dataset_folder, data_npz_file, epoch_train=100, batch_size=8, snippet_len=8, snippet_stride=3,
         model_folder=".", model_file="none.pth"):
    image_array, seq_range_array, cam_offset_array = dataset_reader2(dataset_folder, data_npz_file, force_re=False)

    # 取第一个相机下的所有行人索引作为整个数据集的所有行人索引
    person_ids = list(seq_range_array[:, 0][cam_offset_array[0]:cam_offset_array[1]])

    '''
    # 取第一个相机为probe，第二个相机为gallery，则第二个相机下第一个人的起始帧索引为probe和gallery的分割索引
    # probe_gallery_split_index = seq_range_array[cam_offset_array[1]][1]
    # probe_set = image_array[:probe_gallery_split_index]
    # gallery_set = image_array[probe_gallery_split_index:]
    '''

    # 计算整个数据集上数据的均值和标准差
    '''
    mean_value, std_value = [0.0 for _ in range(3)], [0.0 for _ in range(3)]
    for i in range(3):
        channel_data = image_array[:, :, :, i] / 255
        mean_value[i] = round(np.mean(channel_data), 4)
        std_value[i] = round(np.std(channel_data), 4)
    '''
    mean_value, std_value = [0.4348, 0.4756, 0.3638], [0.1634, 0.1705, 0.1577]

    # 获取分割数据
    splits = get_splits("./prid2011_splits.json", person_ids, shuffle=True, force_re=False)
    transforms = trans.Compose([
        trans.ToTensor(),           # 如果是numpy-ndarray且dtype=unint8会自动除以255，且维度由(h, w, c)变成(c, h, w)
        trans.Normalize(mean=mean_value, std=std_value)
    ])

    device = (torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"))
    # 网络定义
    top_k = (1, 5, 10, 20)
    better_top_k = [.0 for _ in range(len(top_k))]
    net = ResNet(out_features=1000, final_pool="avg")
    old_model_file = osp.join(model_folder, model_file)
    if osp.isfile(old_model_file):
        net.load_state_dict(torch.load(old_model_file))
        print("加载模型参数文件；" + old_model_file)
    else:
        print("未加载任何模型参数文件")
    net = net.cuda(device=device)

    # 损失函数和优化器
    triplet_loss = torch.nn.TripletMarginLoss(margin=1.0, p=2)
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    first_run_check_top_k = True

    for epoch_id in range(epoch_train):
        for split_id, split in enumerate(splits):
            train_ids, test_ids = split["train_ids"], split["test_ids"]
            train_data = train_data_loader2(train_ids, image_array, seq_range_array, cam_offset_array,
                                            batch_size=batch_size, shuffle=True,
                                            snippet_len=snippet_len, snippet_stride=snippet_stride)
            test_data = test_data_loader(test_ids, image_array, seq_range_array, cam_offset_array)

            if first_run_check_top_k:
                mean_top_k = check_top_k(test_data, test_ids, test_ids, transforms=transforms,
                                         net_model=net, device=device, top_k=top_k)
                print("first run, check top-k. top-k: %s" % mean_top_k)
                first_run_check_top_k = False
                if is_better_top_k(old_top_k=better_top_k, new_top_k=mean_top_k):
                    better_top_k = mean_top_k[:]
                    save_model_params(k=top_k, top_k=better_top_k, model_folder=model_folder, model=net)

            running_loss = AverageCalc()
            batch_id = 0
            for batch_id, batch_data in enumerate(train_data):
                anchors_output, positives_output, negatives_output = [], [], []
                for i in range(batch_size):
                    optimizer.zero_grad()
                    anchor_i, positive_i, negative_i = batch_data[i][0], batch_data[i][1], batch_data[i][2]

                    # print(list_shape(anchor_i), list_shape(positive_i), list_shape(negative_i))
                    # draw_4d_list(anchor_i, "anchor", max_cols=4)
                    # draw_4d_list(positive_i, "positive", max_cols=4)
                    # draw_4d_list(negative_i, "negative", max_cols=4)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()
                    # exit(1)

                    # ndarray转变为tensor和数据预处理
                    input_anchor = ndarrays_2_tensor(anchor_i, transforms, grad=True)
                    input_positive = ndarrays_2_tensor(positive_i, transforms, grad=True)
                    input_negative = ndarrays_2_tensor(negative_i, transforms, grad=True)

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

                    anchors_output.append(output_anchor)
                    positives_output.append(output_positive)
                    negatives_output.append(output_negative)

                # 计算损失
                anchors_output = torch.stack(anchors_output, dim=0)
                positives_output = torch.stack(positives_output, dim=0)
                negatives_output = torch.stack(negatives_output, dim=0)

                loss = triplet_loss(anchors_output, positives_output, negatives_output)
                loss.backward()  # 误差反向传播，计算梯度
                optimizer.step()  # 更新权重

                running_loss.update(loss.item())
                # print_process(batch_id+1, batch_num)
                if (batch_id + 1) % 5 == 0:
                    print("epoch: %d \t split: %d \t batch: %d \t avg_loss: %f" %
                          (epoch_id, split_id, batch_id+1, running_loss.value()))
                    running_loss.clear()
            # 数据训练完毕
            if running_loss.count() > 0:
                print("epoch: %d \t split: %d \t batch: %d \t avg_loss: %f" %
                      (epoch_id, split_id, batch_id+1, running_loss.value()))

            # 使用测试集时计算top-k
            mean_top_k = check_top_k(test_data, test_ids, test_ids, transforms=transforms,
                                     net_model=net, device=device, top_k=top_k)
            print("epoch: %d, split: %d, test top-k: %s" % (epoch_id, split_id, mean_top_k))
            # 保存较好的模型
            if is_better_top_k(old_top_k=better_top_k, new_top_k=mean_top_k):
                better_top_k = mean_top_k[:]
                save_model_params(k=top_k, top_k=better_top_k, model_folder=model_folder, model=net)


if __name__ == '__main__':
    main(dataset_folder="/home/haofeng/Desktop/datasets/dst/prid_2011/src",
         data_npz_file="/home/haofeng/Desktop/datasets/dst/prid_2011/src_array.npz",
         epoch_train=10, batch_size=8, snippet_len=8, snippet_stride=3,
         model_folder="../saved_models")
