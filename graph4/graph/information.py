import json

def read_json_data(path):
    """
    读取表示热点区域的json文件
    :param path:     json文件路径
    :return:         热点区域数据
    """

    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

