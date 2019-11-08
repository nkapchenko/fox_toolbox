import functools
import logging
import traceback
import datetime
import reprlib
from itertools import chain, count
from threading import Lock
from os import path

import zlib
import pickle
import atexit


def typed_property(name, expected_type):
    storage_name = '_' + name

    @property
    def prop(self):
        return getattr(self, storage_name)

    @prop.setter
    def prop(self, value):
        if not isinstance(value, expected_type):
            raise TypeError('{} must be a {}'.format(name, expected_type))
        setattr(self, storage_name, value)

    return prop


def quiet_func(func):
    """
    Wrapper function which traces errors and returns None in case of exception.
    Otherwise returns function output.
    """
    @functools.wraps(func)
    def quiet_func_call(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except:
            args_ = str(locals())
            err = traceback.format_exc()
            logging.error(f'{args_} --- {err}')
            return None
    return quiet_func_call


def verbose(print_id=True):
    """
    Wrapper function which prints params of function and enter time
    :param print_id: do you want to print call id
    """
    # thread safe counter
    counter = count()

    def verbose_func(func):
        @functools.wraps(func)
        def verbose_func_call(*args, **kwargs):
            _params = reprlib.repr(tuple(chain(args, kwargs.values())))
            _time = datetime.datetime.today().time()
            _id = next(counter) if print_id else ''
            print('{}.{} --- {}'.format(_id, _time, _params))
            return func(*args, **kwargs)
        return verbose_func_call
    return verbose_func


def get_hash(args, kwargs):
    """
    Get persistent cache key
    """
    params_str = ''.join(map(str, chain(args, kwargs.values()))).encode('utf-8')
    return zlib.crc32(params_str, 0xFFFF)


def load_pickle(fpath):
    """
    Loads python object from a pickle file
    :param fpath: full path to file
    :return: parsed object
    """
    with open(fpath, 'rb') as f:
        return pickle.load(f)


def save_pickle(obj, fpath):
    """
    Saves python object to a pickle file
    :param obj: object to save
    :param fpath: full file path with extension
    """
    with open(fpath, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def memoize(backend_fpath, memoize_none=False, dump_freq=None):
    """
    Wrapper which can be used to memorize results of a function.
    Backend file is used to save and load previously saved results.
    :param backend_fpath: path to a file to save/load cache from
    :param memoize_none: pass true if you want to memoize None outputs
    :param dump_freq: there are chances that dump won't be saved upon crash
    of your program; thus one can set dump frequency to save results upon each
    dump_freq function call
    """
    memoize_lock = Lock()
    counter = count()
    cache = {}
    if path.isfile(backend_fpath):
        try:
            cache = load_pickle(backend_fpath)
        except (IOError, ValueError):
            err = traceback.format_exc()
            logging.error(f'Wasn\'t able to load cache form {backend_fpath} due to: {err}')
            cache = {}

    atexit.register(save_pickle, cache, backend_fpath)

    def memoize_func(func):
        """
        Wrapper function to memorize content of the function
        """
        @functools.wraps(func)
        def verbose_func_call(*args, **kwargs):
            sentinel = object()
            key = get_hash(args, kwargs)
            result = cache.get(key, sentinel)
            if result is not sentinel:
                return result
            result = func(*args, **kwargs)
            with memoize_lock:
                if memoize_none or result is not None:
                    cache[key] = result
                if dump_freq is not None and next(counter) % dump_freq == 0:
                    save_pickle(cache, backend_fpath)
            return result

        return verbose_func_call

    return memoize_func
