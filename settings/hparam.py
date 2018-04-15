import os
import yaml


def load_hparam(filename):
    stream = open(filename, 'r')
    docs = yaml.load_all(stream)
    hparam_dict = dict()
    for doc in docs:
        for k, v in doc.items():
            hparam_dict[k] = v
    return hparam_dict


def merge_dict(user, default):
    if isinstance(user, dict) and isinstance(default, dict):
        for k, v in default.items():
            if k not in user:
                user[k] = v
            else:
                user[k] = merge_dict(user[k], v)
    return user


class Dotdict(dict):
    """
    a dictionary that supports dot notation
    as well as dictionary access notation
    usage: d = DotDict() or d = DotDict({'val1':'first'})
    set attributes: d.val2 = 'second' or d['val2'] = 'second'
    get attributes: d.val2 or d['val2']
    """
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, dct=None):
        dct = dict() if not dct else dct
        for key, value in dct.items():
            if hasattr(value, 'keys'):
                value = Dotdict(value)
            self[key] = value


class Hparam(Dotdict):

    def __init__(self):
        super(Dotdict, self).__init__()

    __getattr__ = Dotdict.__getitem__
    __setattr__ = Dotdict.__setitem__
    __delattr__ = Dotdict.__delitem__
    DIR_PATH = os.path.dirname(__file__)
    DEFAULT_FILE = os.path.join(DIR_PATH, '..', 'hparams/default.yaml')
    USER_FILE = os.path.join(DIR_PATH, '..', 'hparams/hparams.yaml')

    def set_hparam_yaml(self, case, default_file=DEFAULT_FILE, user_file=USER_FILE):
        default_hp = load_hparam(default_file)
        user_hp = load_hparam(user_file)
        hp_dict = Dotdict(merge_dict(user_hp[case], default_hp) if case in user_hp else default_hp)
        for k, v in hp_dict.items():
            setattr(self, k, v)
        self._auto_setting(case)
        print(getattr(self, 'default'))

    def _auto_setting(self, case):
        setattr(self, 'case', case)
        setattr(self, 'logdir', '{}/{}'.format(hparam.logdir_path, case))


hparam = Hparam()