from utils.service_helpers import get_dic_item


def get_json_ccy(task):
    node = get_dic_item(task, 'marketData/rates')
    keys = list((lbl for lbl in node.keys() if len(lbl) == 3))
    if len(keys) > 1:
        raise NotImplementedError('More than 1 ccy')
    return keys[0]


def get_json_equity(task):
    node = get_dic_item(task, 'marketData/equities')
    keys = list(node.keys())
    if len(keys) > 1:
        raise NotImplementedError('More than 1 equity process')
    return keys[0]