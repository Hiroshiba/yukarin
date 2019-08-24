import argparse
import glob
import multiprocessing
from pathlib import Path
from pprint import pprint

import numpy
import tqdm

from yukarin.acoustic_feature import AcousticFeature

parser = argparse.ArgumentParser()
parser.add_argument('--input_glob', '-i')
parser.add_argument('--output', '-o', type=Path)
arguments = parser.parse_args()


def load_f0(path: Path):
    feature = AcousticFeature.load(path=path)
    return feature.f0


def main():
    pprint(vars(arguments))

    paths = [Path(p) for p in sorted(glob.glob(arguments.input_glob))]

    pool = multiprocessing.Pool()
    it = pool.imap(load_f0, paths)
    f0_list = list(tqdm.tqdm(it, total=len(paths)))

    f0 = numpy.concatenate(f0_list)
    f0 = f0[f0.nonzero()]
    log_f0 = numpy.log(f0)

    mean, var = log_f0.mean(), log_f0.var()
    numpy.save(arguments.output, dict(mean=mean, var=var))


if __name__ == '__main__':
    main()
