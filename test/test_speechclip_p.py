import argparse
import sys

# insert at 1, 0 is the script path (or '' in REPL)
# sys.path.insert(1, '/work/vjsalt22/hsuanfu/audio-visual-ssl')
from avssl.model import ParallelSpeechClip


def test_speechclip_p():
    args = argparse.Namespace(
        seed=7122,
        config="config/speechclip_p/train_flickr.yaml",
        device="cuda:0",
        gpus=1,
        ckpt="",
        njobs=2,
        save_path="exp/sc_p_tmp",
    )

    ParallelSpeechClip(args)
