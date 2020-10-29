import random


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


if __name__ == '__main__':
    data = list(range(10))
    for i in range(100):
        print(list_choice(data_list=data, exclude=0))
