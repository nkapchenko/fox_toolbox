from os import path
import json
import reprlib
import requests


def pprint_json(task):
    """
    Prints json with 3 spaces indent
    :param task: json task to print
    """
    print(json.dumps(task, indent=3))


def pprint_succinct(_dic, maxlvl=4, maxlen=80):
    """
    Pretty-prints dictionaries and jsons limiting output amount
    :param _dic: json to print
    :param maxlvl: maximum depth to print; 4 by default
    :param maxlen: maximum width of output; 80 chars by default
    """
    myrepr = reprlib.Repr()
    myrepr.maxstring = maxlen
    myrepr.maxlevel = maxlvl
    print('{')
    for k, v in _dic.items():
        print(f'    \'{k}\': {myrepr.repr(v)},')
    print('}')


def split_path(xpath):
    return xpath.strip('/').split('/')


def get_dic_item(_dic, xpath):
    """
    Returns item of multilevel dictionary as json by provided xpath
    :param _dic: dictionary to search
    :param xpath: tab delimited path as 'rate/USD/...'
    :return: found element or None if key doesn't exist
    """
    elem = _dic
    try:
        for x in split_path(xpath):
            elem = elem.get(x)
        return elem
    except KeyError:
        return None


def set_dic_item_safe(_dic, xpath, value):
    """
    Set's node of multilevel dictionary to a provided value.
    If key doesn't exists will insert new node.
    :param _dic: dictionary to set value
    :param xpath: xpath of node like 'key1/key2/key3'
    :param value: value
    """
    elem = _dic
    nodes = split_path(xpath)
    try:
        for x in nodes[:-1]:
            elem = elem.get(x)
        elem[nodes[-1]] = value
    except KeyError:
        if len(nodes) == 1:
            elem[xpath] = value


def set_dic_item(_dic, xpath, value):
    """

    :param _dic:
    :param xpath:
    :param value:
    :return:
    """
    elem = _dic
    nodes = split_path(xpath)
    if len(nodes) == 1:
        elem[xpath] = value
    else:
        for x in nodes[:-1]:
            elem = elem.setdefault(x, {})
        elem[nodes[-1]] = value


def save_json(task, fpath):
    with open(fpath, 'w') as file_:
        json.dump(task, file_, indent=4)


def load_json(fpath):
    with open(fpath, 'r') as file_:
        return json.load(file_)


def download_url(url, directory=None, fname=None, extension=None, session=None):
    """
    Downloads content of the url
    :param url: url path
    :param directory: directory to save file
    :param fname: file name which will be joined with directory name
    :param extension: file extension
    :param session: Open session of requests.Session type
    :return: full file path to downloaded file
    """
    if fname is None:
        fname = url.split('/')[-1]
    if extension:
        fname += extension
    fpath = path.join(directory, fname) if directory else fname

    if session is None:
        session = requests.Session()
    with session.get(url, stream=True) as response:
        response.raise_for_status()
        with open(fpath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    return fpath


def download_traces(qres, fldr, fprefix):
    """
    Downloads all traces from a query result in provided folder
    :param qres: service query result dictionary
    :param fldr: folder where to save traces
    :param fprefix: prefix to add to file names of traces
    """
    for i, url in enumerate(qres['traceAddress']):
        with requests.Session() as session:
            with session.get(url.strip()) as response:
                data = json.loads(response.text)
                for trace, trace_url in data['files'].items():
                    extension = '.json' if trace == 'payload' else '.xml'
                    download_url(trace_url, directory=fldr, fname=f'{fprefix}_{trace}_{i}',
                                 extension=extension, session=session)
