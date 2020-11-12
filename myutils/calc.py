import time


class AverageCalc:
    def __init__(self):
        self.__count = 0
        self.__value = 0.0

    def update(self, v):
        self.__value = self.__value + (v - self.__value) / (self.__count + 1)
        self.__count += 1

    def clear(self):
        self.__count = 0
        self.__value = 0.0

    def value(self):
        return self.__value

    def count(self):
        return self.__count


class LoopTimeCalc:
    """统计循环耗时"""
    def __init__(self):
        self.__avgCalc = AverageCalc()
        self.__last_time = time.time()

    def restart(self):
        self.__avgCalc.clear()
        self.__last_time = time.time()

    def update(self):
        cur_time = time.time()
        diff_time = cur_time - self.__last_time
        self.__avgCalc.update(diff_time)
        self.__last_time = cur_time

    # def clear(self):
    #     self.__avgCalc.clear()
    #     self.__last_time = time.time()

    def avg_time(self):
        return self.__avgCalc.value()


if __name__ == '__main__':
    avg = AverageCalc()
    data_list = []
    for i in range(10):
        data_list.append(i)
        avg.update(i)
        print(data_list, "\t----\t", avg.value())
    data_list.clear()
    avg.clear()
    for i in range(10, 15):
        data_list.append(i)
        avg.update(i)
        print(data_list, "\t----\t", avg.value())
