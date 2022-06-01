
from avssl.model import speechclip_c as mymodels
from avssl.model.speechclip_c import *

import argparse
import os
import sys
import zerospeech_tasks
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def parseArgs(argv):
    # Run parameters
    parser = argparse.ArgumentParser(description='Export BERT features from quantized units of audio files.')
    parser.add_argument("--model_cls_name",type=str)
    parser.add_argument("--model_ckpt",type=str)
    parser.add_argument("--task_input_dir",type=str,default="/work/{}/dataset/zerospeech2021/semantic/".format(os.environ.get("USER")))
    parser.add_argument("--task_name",type=str,default="semantic")
    parser.add_argument("--output_result_dir",type=str)
    parser.add_argument(
            "--run_dev", action="store_true", default=False, help="run dev"
        )
    parser.add_argument(
            "--run_test", action="store_true", default=False, help="run dev"
        )

    return parser.parse_args(argv)

def loadModel(_cls,_path):
    _model = getattr(mymodels,_cls).load_from_checkpoint(_path)
    return _model

def main(argv):
    args = parseArgs(argv)
    # print(args.__dict__) 
    task_cls = f"Task_{args.task_name}"

    assert hasattr(zerospeech_tasks,task_cls)

    assert hasattr(mymodels,args.model_cls_name )

    logger.info(f"Loading model({args.model_cls_name}) from {args.model_ckpt}")
    mymodel = loadModel(args.model_cls_name,args.model_ckpt)
    mytask = getattr(zerospeech_tasks,task_cls)(**args.__dict__,my_model=mymodel)

    mytask.run()

if __name__ == "__main__":
    args = sys.argv[1:]
    main(args)