# 文件操作

import json
import numpy as np


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


def write_json_file(data_dict, json_file_path):
    with open(json_file_path, 'w') as fp:
        json.dump(data_dict, fp=fp, indent=4, cls=MyEncoder)


def read_json_file(json_file_path):
    return json.load(open(json_file_path, 'r'))
