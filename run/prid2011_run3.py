# -*- coding: utf-8 -*-

import random
import os.path as osp
import json
import cv2
import time

import torch
from torch import optim
import torchvision.transforms as trans

from model.resnet import ResNet
from myutils.func import *
from myutils.listutils import list_sort
from myutils.calc import AverageCalc, LoopTimeCalc
from data.dataset.dataloader2 import dataset_reader, train_data_loader, test_data_loader, get_split
from model.dist_aggregate import dist_aggr_top_avg


@torch.no_grad()
def fvs_dist(p_snippets_fvs, g_snippets_fvs):
    dists = []
    for p_fv in p_snippets_fvs:
        for g_fv in g_snippets_fvs:
            p = torch.from_numpy(p_fv)
            p = torch.unsqueeze(p, dim=0)

            g = torch.from_numpy(g_fv)
            g = torch.unsqueeze(g, dim=0)

            dists.append(torch.pairwise_distance(p, g).item())

    return dist_aggr_top_avg(dists, top_rank=0.2)


@torch.no_grad()
def check_top_k(test_data, probe_ids, gallery_ids, transforms, net_model, device, top_k=(1, 5, 10, 20)):
    print("top-k 计算中...")
    p_snippets, g_snippets = test_data[0], test_data[1]
    p_person_num, g_person_num = len(p_snippets), len(g_snippets)

    # 首先每个片段经过网络生成对应的特征向量
    probe_fvs, gallery_fvs = [], []
    for snippets in p_snippets:
        s_fvs = []
        for snippet in snippets:
            s_input = ndarrays_2_tensor(snippet, transforms, grad=False)
            s_input = s_input.to(device)
            s_output = net_model(s_input)
            s_fvs.append(s_output.cpu().numpy())
        probe_fvs.append(s_fvs)

    for snippets in g_snippets:
        s_fvs = []
        for snippet in snippets:
            s_input = ndarrays_2_tensor(snippet, transforms, grad=False)
            s_input = s_input.to(device)
            s_output = net_model(s_input)
            s_fvs.append(s_output.cpu().numpy())
        gallery_fvs.append(s_fvs)

    # 计算probe和gallery中每个人的序列距离或相似度（序列的距离或相似度由片段集的距离或相似度融合得到）
    dist_matrix = np.zeros(shape=(p_person_num, g_person_num), dtype=np.float)
    for i in range(p_person_num):
        for j in range(g_person_num):
            dist_matrix[i][j] = fvs_dist(probe_fvs[i], gallery_fvs[j])

    # 获得按距离由小到大或相似度由大到小的gallery-label二维数组，一维为对应的probe下标
    pid_matrix = []
    for i in range(p_person_num):
        pid_matrix.append(list_sort(gallery_ids, dist_matrix[i], "ASC"))

    # 计算每个probe的top-k
    top_k_matrix = np.zeros(shape=(p_person_num, len(top_k)), dtype=np.int)
    for i in range(p_person_num):
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
    filename = "model_params"
    for i in range(len(k)):
        filename += ("_top%d_%.4f" % (k[i], top_k[i]))
    filename += ".pth"
    return filename


def save_model_params(k, top_k, model_folder, model):
    mkdir_if_missing(model_folder)
    filename = get_model_params_file_name(k, top_k)
    filepath = osp.join(model_folder, filename)
    torch.save(model.state_dict(), filepath)
    print("保存模型参数到：" + filepath)


def main(dataset_folder, data_mode="src", image_suffix=".png", epoch_train=10,
         split_id=0, split_file="", person_num=200, train_rate=0.5, force_re_split=False,
         batch_size=8, snippet_len=8, snippet_stride=4,
         model_folder=".", model_file="none.pth"):

    images_array, seq_range_array = dataset_reader(dataset_folder, data_mode, image_suffix)
    print(images_array.shape, seq_range_array.shape)

    # 计算整个数据集上各通道数据的均值和标准差
    channel_num = images_array.shape[-1]
    mean_value, std_value = [0.0 for _ in range(channel_num)], [0.0 for _ in range(channel_num)]
    for i in range(channel_num):
        channel_data = images_array[:, :, :, i] / 255
        mean_value[i] = round(np.mean(channel_data), 4)
        std_value[i] = round(np.std(channel_data), 4)
    print("mean: %s\t std: %s" % (str(mean_value), str(std_value)))
    # mean_value, std_value = [0.4348, 0.4756, 0.3638], [0.1634, 0.1705, 0.1577]
    # mean_value, std_value = [0.0397, 0.0371, 0.0279], [0.1884, 0.1796, 0.1586]

    # 获取分割数据
    split = get_split(split_id, split_file=split_file,
                      person_num=person_num, train_rate=train_rate, force_re=force_re_split)
    train_ids, test_ids = split["train_ids"], split["test_ids"]
    print("split_id=%d" % split["split_id"])

    # 定义数据变换
    transforms = trans.Compose([
        trans.ToTensor(),           # 如果是numpy-ndarray且dtype=unint8会自动除以255，且维度由(h, w, c)变成(c, h, w)
        trans.Normalize(mean=mean_value, std=std_value)
    ])

    # GPU设备
    device = (torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"))
    # 网络定义
    net = ResNet(in_channels=channel_num, out_features=1000, final_pool="avg")
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

    top_k = (1, 5, 10, 20)
    better_top_k = [0.0 for _ in range(len(top_k))]
    test_data = test_data_loader(test_ids, images_array, seq_range_array,
                                 snippet_len=snippet_len, snippet_stride=snippet_stride)
    running_loss = AverageCalc()

    for epoch_id in range(epoch_train):
        train_data = train_data_loader(train_ids, images_array, seq_range_array,
                                       batch_size=batch_size, shuffle=True,
                                       snippet_len=snippet_len, snippet_stride=snippet_stride)
        # 首次训练需要测试一下初始模型的效果
        if epoch_id == 0:
            mean_top_k = check_top_k(test_data, test_ids, test_ids, transforms=transforms,
                                     net_model=net, device=device, top_k=top_k)
            print("initial model, top-k: %s" % mean_top_k)
            if is_better_top_k(old_top_k=better_top_k, new_top_k=mean_top_k):
                better_top_k = mean_top_k[:]
                save_model_params(k=top_k, top_k=better_top_k, model_folder=model_folder, model=net)

        running_loss.clear()
        batch_id = 0
        for batch_id, batch_data in enumerate(train_data):
            optimizer.zero_grad()
            anchors_output, positives_output, negatives_output = [], [], []
            for i in range(batch_size):
                input_data, input_label = batch_data[i][0], batch_data[i][1]
                anchor_i, positive_i, negative_i = input_data[0], input_data[1], input_data[2]

                # ndarray转变为tensor和数据预处理
                input_anchor = ndarrays_2_tensor(anchor_i, transforms, grad=True)
                input_positive = ndarrays_2_tensor(positive_i, transforms, grad=True)
                input_negative = ndarrays_2_tensor(negative_i, transforms, grad=True)

                # 转移到cuda上
                input_anchor = input_anchor.to(device)
                input_positive = input_positive.to(device)
                input_negative = input_negative.to(device)

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
            if (batch_id + 1) % 10 == 0:
                print("epoch: %d \t batch: %d \t avg_loss: %f" % (epoch_id, batch_id+1, running_loss.value()))
                running_loss.clear()
        # 数据训练完毕
        if running_loss.count() > 0:
            print("epoch: %d \t batch: %d \t avg_loss: %f" % (epoch_id, batch_id+1, running_loss.value()))

        # 使用测试集时计算top-k
        mean_top_k = check_top_k(test_data, test_ids, test_ids, transforms=transforms,
                                 net_model=net, device=device, top_k=top_k)
        print("epoch: %d, test top-k: %s" % (epoch_id, mean_top_k))
        # 保存较好的模型
        if is_better_top_k(old_top_k=better_top_k, new_top_k=mean_top_k):
            better_top_k = mean_top_k[:]
            save_model_params(k=top_k, top_k=better_top_k, model_folder=model_folder, model=net)


if __name__ == '__main__':
    folder_dataset = "/home/haofeng/Desktop/datasets/dst/prid_2011"
    file_split = "/home/haofeng/Desktop/datasets/dst/prid_2011/prid2011_splits.json"
    # train_data_mode = "src"
    # train_data_mode = "pose"
    train_data_mode = "src_pose"
    split_index = 0
    folder_model_params = "/home/haofeng/Desktop/saved_models/prid2011_split{}_{}".format(split_index, train_data_mode)

    main(dataset_folder=folder_dataset,
         data_mode=train_data_mode, epoch_train=20,
         split_id=0, split_file=file_split, force_re_split=False,
         batch_size=8, snippet_len=8, snippet_stride=4,
         model_folder=folder_model_params,
         model_file="model_params_top1_0.5500_top5_0.8600_top10_0.9200_top20_0.9800.pth")
