import numpy as np
import pickle

freq_usage = np.load("./text_clip_vocab_usage_byfreq.npy")
ID_usage = np.load("./text_clip_vocab_usage_byID.npy")
print(ID_usage.shape)
res = {}

for i, row in enumerate(freq_usage):
    id, freq = row[0], row[1]
    if id == 1081:
        print(freq)
    res[id] = i

for i, row in enumerate(ID_usage):
    id, freq = row[0], row[1]
    if id == 1081:
        print(freq)

# write
with open('token_mapping.p', 'wb') as fp:
    pickle.dump(res, fp)

# read
with open('token_mapping.p', 'rb') as fp:
    data = pickle.load(fp)