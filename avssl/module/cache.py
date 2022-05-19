from typing import Any, List

import torch


class SimpleCache:
    def __init__(self, cache_keys: List[Any]) -> None:
        self.memory_dict = {key: dict() for key in cache_keys}

    def has_key(self, c_key: Any) -> bool:
        return c_key in self.memory_dict

    def add_key(self, c_key: Any) -> None:
        if c_key not in self.memory_dict:
            self.memory_dict[c_key] = dict()

    def exists(self, c_key: Any, i_key: Any) -> bool:
        if c_key not in self.memory_dict:
            return False
        return i_key in self.memory_dict[c_key]

    def exists_list(self, c_key: Any, i_key_list: List[Any]) -> bool:
        return all(self.exists(c_key, i_key) for i_key in i_key_list)

    def save(self, c_key: Any, i_key: Any, value) -> None:
        if self.exists(c_key, i_key):
            return
        if c_key not in self.memory_dict:
            raise KeyError(f"c_key {c_key} not in memory")

        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().pin_memory()

        self.memory_dict[c_key][i_key] = value

    def get(self, c_key: Any, i_key: Any, value=None) -> Any:
        if not self.exists(c_key, i_key):
            return value
        return self.memory_dict[c_key][i_key]

    def get_batch(self, c_key: Any, i_key_list: List[Any]) -> Any:
        data = []
        for i_key in i_key_list:
            data.append(self.memory_dict[c_key][i_key])
        return data
