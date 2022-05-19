class EmbeddingCache:
    def __init__(self, keys):
        self.data_dict = {k: [] for k in keys}
        self.data_keys = keys

    def clear_cache(self):
        del self.data_dict
        self.data_dict = {k: [] for k in self.data_keys}

    def add(self, _key, _data):
        self.data_dict[_key].append(_data)

    def get_data(self, _key):
        return self.data_dict[_key]
