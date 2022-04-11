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

probs = _data[:, 1] / np.sum(_data[:, 1])

mock_prob = np.ones((8112), dtype=float) / 8112


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
