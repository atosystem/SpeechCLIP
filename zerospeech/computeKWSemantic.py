import os
import pathlib
import numpy as np
import yaml
from zerospeech2021 import semantic
import sys
import pandas as pd


DATASET_DIR = "/work/twsezjg982/dataset/zerospeech2021"
N_JOBS = 16
KW_NUM = 16

def eval_semantic(dataset, submission, kinds, njobs):
    # load metric and poling parameters from meta.yaml
    meta = yaml.safe_load((submission / 'meta.yaml').open('r').read())
    metric = meta['parameters']['semantic']['metric']
    pooling = meta['parameters']['semantic']['pooling']


    all_results = {
        "kw" : [],
        "librispeech" : [],
        "synthetic" : [],
    }

    for kw_i in range(KW_NUM):
        pooling = "kw_{}".format(kw_i)
        print(pooling)
        for kind in kinds:  # 'dev' or 'test'
            print(f'Evaluating semantic {kind} '
                f'(metric={metric}, pooling={pooling})...')

            gold_file = dataset / 'semantic' / kind / 'gold.csv'
            pairs_file = dataset / 'semantic' / kind / 'pairs.csv'
            pairs, correlation = semantic.evaluate(
                gold_file, pairs_file, submission / 'semantic' / kind,
                metric, pooling, njobs=njobs)
            if kind == "dev":
                all_results["kw"].append(kw_i)
                for index, row in correlation.iterrows():
                    all_results[row["type"]].append(row["correlation"])
        
        assert len(all_results['kw']) == len(all_results['librispeech'])
        assert len(all_results['kw']) == len(all_results['synthetic'])

    df = pd.DataFrame(all_results)

    return df



if __name__ == "__main__":
    assert len(sys.argv) == 2
    exp_dir = sys.argv[1]

    example_data = np.loadtxt(os.path.join(exp_dir,"semantic/dev/librispeech/admrqDlxdu.txt"))

    print("Data shape : {}".format(example_data.shape))

    KW_NUM = example_data.shape[0]

    dataset = pathlib.Path(DATASET_DIR)
    submission = pathlib.Path(exp_dir)

    dataset = dataset.resolve(strict=True)
    if not dataset.is_dir():
        raise ValueError(f'dataset not found: {dataset}')

    submission = submission.resolve(strict=True)

    correlation_df = eval_semantic(dataset, submission, ["dev"], N_JOBS)


    correlation_df.to_csv(
        os.path.join(exp_dir,"semantic_score.csv"),
        index=False
    )
    print(correlation_df)

