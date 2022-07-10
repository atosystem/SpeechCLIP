import numpy as np
import pandas as pd
import plotly.express as px
import torch
from sklearn.decomposition import PCA


def draw_embedding_space_PCA(
    kw_embs: torch.Tensor, gold_embs: torch.Tensor, output_path: str
):
    kw_count = kw_embs.shape[1]
    kw_embs = kw_embs.reshape(-1, gold_embs.shape[-1])
    kw_embs = kw_embs.numpy()
    gold_embs = gold_embs.numpy()

    _data = {
        "token_emb": [gold_embs[i] for i in range(gold_embs.shape[0])]
        + [kw_embs[i] for i in range(kw_embs.shape[0])],
        "type": ["pretrained"] * gold_embs.shape[0] + ["kw"] * kw_embs.shape[0],
        "kw_id": ["pretrained"] * gold_embs.shape[0]
        + ["kw_{}".format(x) for x in range(kw_count)] * (kw_embs.shape[0] // kw_count),
    }

    df = pd.DataFrame(_data, index=None)

    pca = PCA()

    components = pca.fit_transform(np.concatenate((gold_embs, kw_embs), axis=0))

    fig = px.scatter(components, x=0, y=1, color=df["kw_id"])
    # fig.show()

    # labels = {
    #     str(i): f"PC {i+1} ({var:.1f}%)"
    #     for i, var in enumerate(pca.explained_variance_ratio_ * 100)
    # }

    # fig = px.scatter_matrix(
    #     components, labels=labels, dimensions=range(2), color=df["kw_id"]
    # )
    # fig.update_traces(diagonal_visible=False)

    fig.write_image(output_path)
