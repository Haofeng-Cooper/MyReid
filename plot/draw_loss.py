import os.path as osp
import matplotlib.pyplot as plt


def plot_file(filepath):
    loss_list = []
    if not osp.isfile(filepath):
        raise ValueError("{}不是一个文件".format(filepath))

    with open(filepath, 'r') as fp:
        while True:
            line_text = fp.readline().strip()
            if not line_text:
                break
            str_list = line_text.split('\t')
            if len(str_list) == 3:
                loss_value = str_list[-1].split(' ')[-1]
                loss_list.append(float(loss_value))

    print(len(loss_list))

    x = list(range(len(loss_list)))
    plt.plot(x, loss_list)
    plt.show()


if __name__ == '__main__':
    # plot_file("/home/haofeng/Desktop/saved_models/prid2011_pose_split0/log.txt")
    plot_file("/home/haofeng/Desktop/saved_models/prid2011_src_split0/log.txt")
