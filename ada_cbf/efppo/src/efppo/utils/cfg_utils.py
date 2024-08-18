from attrs import asdict, astuple

from argparse import Namespace

class Cfg:
    def asdict(self):
        return asdict(self)

    def astuple(self):
        return astuple(self)


 
class RecursiveNamespace(Namespace):
    def __init__(self, **kwargs):
        self.kwargs= kwargs
        self.sub_spaces = list()
        self.name_lst = list()

        dict_kwargs = {}
        name_kwargs = {}
        for name in kwargs:
            if type(kwargs[name]) == dict:
                dict_kwargs[name] = kwargs[name]
                self.sub_spaces.append(name)
            else:
                name_kwargs[name] = kwargs[name]
                self.name_lst.append(name)
        
        super().__init__(**name_kwargs)
        
        for name in dict_kwargs:
            setattr(self, name, RecursiveNamespace(**dict_kwargs[name]))
            self.sub_spaces.append(name)
    
    def update(self, namespace):
        for name in namespace.name_lst:
            setattr(self, name, getattr(namespace, name))

    def copy(self):
        return RecursiveNamespace(**self.kwargs)