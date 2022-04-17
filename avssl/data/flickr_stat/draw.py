import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import entropy
from torch.nn import functional as F


def KL(a, b):
    a = np.asarray(a, dtype=np.float)
    b = np.asarray(b, dtype=np.float)

    return np.sum(np.where(a != 0, a * np.log(a / b), 0))


_data = np.load("text_clip_vocab_usage_byID.npy")

# print(_data[-10:,:])
# exit(1)
# _data = _data[1:-2,:]
# [[      0 2581899]
#  [    320   62996]
#  [  49406   40460]
#  [  49407

probs = _data[:, 1] / np.sum(_data[:, 1])

mock_prob = np.ones((len(probs)), dtype=float) / len(probs)


ent = entropy(probs)
perplexity = np.exp(ent)

print("entropy : {}".format(ent))
print("perplexity : {}".format(perplexity))

print(
    "KL with uniform",
    F.kl_div(torch.tensor(mock_prob).log(), torch.tensor(probs), reduction="sum"),
)

plt.plot(_data[:, 1])
plt.savefig("vocab_usage_byID.png")

# entropy : 1.3691492735673139
# perplexity : 3.9320042122846286
# KL with uniform tensor(7.6320, dtype=torch.float64)
