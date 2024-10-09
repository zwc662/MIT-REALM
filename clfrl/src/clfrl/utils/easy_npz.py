import pathlib
import pickle
from typing import Any, Mapping, overload

import numpy as np

na = np.ndarray
na2 = tuple[na, na]
na3 = tuple[na, na, na]
na4 = tuple[na, na, na, na]
na5 = tuple[na, na, na, na, na]
na6 = tuple[na, na, na, na, na, na]
na7 = tuple[na, na, na, na, na, na, na]
na8 = tuple[na, na, na, na, na, na, na, na]
na9 = tuple[na, na, na, na, na, na, na, na, na]
na10 = tuple[na, na, na, na, na, na, na, na, na, na]


class EasyNpz(Mapping[str, np.ndarray]):
    """Wrapper for npz that provides a shorthand for accessing multiple keys."""

    def __init__(self, npz_or_pkl_path: pathlib.Path):
        self.dictlike: dict[str, Any] = {}
        if npz_or_pkl_path.suffix == ".pkl":
            with open(npz_or_pkl_path, "rb") as f:
                self.dictlike = pickle.load(f)
        elif npz_or_pkl_path.suffix == ".npz":
            self.dictlike = np.load(npz_or_pkl_path)
        else:
            raise ValueError(f"Unknown file extension {npz_or_pkl_path.suffix}")

    @overload
    def __call__(self, k1: str) -> na:
        ...

    @overload
    def __call__(self, k1: str, k2: str) -> na2:
        ...

    @overload
    def __call__(self, k1: str, k2: str, k3: str) -> na3:
        ...

    @overload
    def __call__(self, k1: str, k2: str, k3: str, k4: str) -> na4:
        ...

    @overload
    def __call__(self, k1: str, k2: str, k3: str, k4: str, k5: str) -> na5:
        ...

    @overload
    def __call__(self, k1: str, k2: str, k3: str, k4: str, k5: str, k6: str) -> na6:
        ...

    @overload
    def __call__(self, k1: str, k2: str, k3: str, k4: str, k5: str, k6: str, k7: str) -> na7:
        ...

    @overload
    def __call__(self, k1: str, k2: str, k3: str, k4: str, k5: str, k6: str, k7: str, k8: str) -> na8:
        ...

    @overload
    def __call__(self, k1: str, k2: str, k3: str, k4: str, k5: str, k6: str, k7: str, k8: str, k9: str) -> na9:
        ...

    @overload
    def __call__(
        self, k1: str, k2: str, k3: str, k4: str, k5: str, k6: str, k7: str, k8: str, k9: str, k10: str
    ) -> na10:
        ...

    def __call__(self, *args: str) -> np.ndarray | list[np.ndarray]:
        args = list(args)
        out = [self.dictlike[arg] for arg in args]
        if len(out) == 1:
            return out[0]
        return out

    def __getitem__(self, item):
        return self.dictlike[item]

    def keys(self):
        return self.dictlike.keys()

    def __len__(self):
        return len(self.dictlike)

    def __iter__(self):
        return self.dictlike.__iter__()


def save_data(path: pathlib.Path, **kwargs):
    assert path.suffix == ".pkl"
    data = dict(**kwargs)
    with open(path, "wb") as f:
        pickle.dump(data, f)
