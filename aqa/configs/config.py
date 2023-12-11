from itertools import chain
import json


class Config(dict):
    def __init__(self, __map_or_iterable=None, **kwargs):
        if __map_or_iterable is None:
            d = {}
        else:
            d = dict(__map_or_iterable)

        if kwargs:
            d.update(**kwargs)

        for k, v in d.items():
            setattr(self, k, v)

        # Class attributes
        for k in self.__class__.__dict__.keys():
            if (
                not (k.startswith("__") and k.endswith("__"))
                and k not in ["update", "update_not_recursive", "pop"]
            ):
                setattr(self, k, getattr(self, k))

    def __setattr__(self, __name, __value):
        if isinstance(__value, (list, tuple)):
            __value = [self.__class__(**x) if isinstance(x, dict) else x
                       for x in __value]
        elif isinstance(__value, dict):
            __value = self.__class__(**__value)

        super(Config, self).__setattr__(__name, __value)
        super(Config, self).__setitem__(__name, __value)

    __setitem__ = __setattr__

    def update(self, __map_or_iterable=None, **kwargs):
        __d = self.__class__(__map_or_iterable)

        for k, v in chain(__d.items(), kwargs.items()):
            if isinstance(getattr(self, k, None), self.__class__):
                getattr(self, k).update(v)
            else:
                setattr(self, k, v)

    def update_not_recursive(self, __map_or_iterable=None, **kwargs):
        __d = self.__class__(__map_or_iterable)

        for k, v in chain(__d.items(), kwargs.items()):
            setattr(self, k, v)

    def pop(self, __key, __default=None):
        delattr(self, __key)
        return super(Config, self).pop(__key, __default)

    def __str__(self):
        return json.dumps(self, indent=" " * 4)
