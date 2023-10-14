import os
import argparse
from hxinfer import utils


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ngf-1", type=str, help="the source ngf 1.")
    parser.add_argument("--ngf-2", type=str, help="the source ngf 2.")
    parser.add_argument("--output", type=str, default="", help="the output of combine ngf.")

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    ngf_1 = args.ngf_1
    ngf_2 = args.ngf_2
    assert os.path.exists(ngf_1), f"{ngf_1} is not exists, please check your ngf model path."
    assert os.path.exists(ngf_2), f"{ngf_2} is not exists, please check your ngf model path."
    combine_ngf = args.output
    if combine_ngf == "":
        combine_ngf = "./multi.ngf"
    utils.combine_engine_file([ngf_1, ngf_2], combine_ngf)
