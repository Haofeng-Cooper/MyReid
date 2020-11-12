from math import ceil
from myutils.calc import AverageCalc
import random


def dist_aggr_top_avg(dist_list, top_rank=0.2):
    """
    距离融合，取距离最小的top_rank的比例的均值
    """
    top_num = max(1, ceil(len(dist_list) * top_rank))
    dist_list = sorted(dist_list)

    avg = AverageCalc()
    for i in range(top_num):
        avg.update(dist_list[i])
    return round(avg.value(), 6)


if __name__ == '__main__':
    scores = [random.randint(1, 5) for _ in range(50)]
    print(dist_aggr_top_avg(scores, top_rank=0.5))
