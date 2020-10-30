import random
import numpy as np


def list_choice(data_list: list, exclude: int):
    """
    从数据列表中随机选一个出来，但是元素的值不能等于exclude

    :param data_list: 数据列表
    :param exclude: 随机取的元素值不能等于该值
    """
    rand_value = random.choice(data_list)
    if exclude is not None:
        while rand_value == exclude:
            rand_value = random.choice(data_list)
    return rand_value


def list_relist(data_list, index_list):
    """
    按 index_list 对 data_list 进行重新排列
    """
    result = []
    for index in index_list:
        result.append(data_list[index])
    return result


def list_sort(want_order_list, order_by_list, order_way="ASC"):
    """
    按 order_by_list 中元素的大小顺序，对want_order_list中的元素进行排序。
    want_order_list 中的元素与 order_by_list 中的元素一一对应

    :param want_order_list: 要排序的列表 1-d
    :param order_by_list: 排序的依据列表 1-d
    :param order_way: 升序为 ASC，降序为 DESC
    """

    want_order_list = np.asarray(want_order_list)
    order_by_list = np.asarray(order_by_list)

    if order_way == "DESC":
        order_by_list = -order_by_list

    ordered_index_list = np.argsort(order_by_list)
    ordered_list = list_relist(want_order_list, ordered_index_list)
    return ordered_list


if __name__ == '__main__':
    # data = list(range(10))
    # for i in range(100):
    #     print(list_choice(data_list=data, exclude=0))

    data = list(range(100, 115))
    random.shuffle(data)
    print(data)
    order_list = list(range(15))
    random.shuffle(order_list)
    print(order_list)
    print(list_sort(data, order_list, "DESC"))
