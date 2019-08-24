"""
extract indexes for alignment.
"""

import argparse
import glob
import multiprocessing
from pathlib import Path
from pprint import pprint
from typing import Tuple

import tqdm

from yukarin.acoustic_feature import AcousticFeature
from yukarin.align_indexes import AlignIndexes
from yukarin.utility.json_utility import save_arguments

parser = argparse.ArgumentParser()
parser.add_argument('--input_glob1', '-i1')
parser.add_argument('--input_glob2', '-i2')
parser.add_argument('--output', '-o', type=Path)
parser.add_argument('--dtype', type=str, default='int32')
parser.add_argument('--ignore_feature', nargs='+', default=('feature1', 'feature2'))
parser.add_argument('--enable_overwrite', action='store_true')
arguments = parser.parse_args()


def generate_align_indexes(pair_path: Tuple[Path, Path]):
    path1, path2 = pair_path
    if path1.stem != path2.stem:
        print('warning: the file names are different', path1, path2)

    out = Path(arguments.output, path1.stem + '.npy')
    if out.exists() and not arguments.enable_overwrite:
        return

    feature1 = AcousticFeature.load(path=path1)
    feature2 = AcousticFeature.load(path=path2)

    align_indexes = AlignIndexes.extract(feature1, feature2, dtype=arguments.dtype)

    # save
    align_indexes.save(path=out, ignores=arguments.ignore_feature)


def main():
    pprint(vars(arguments))

    arguments.output.mkdir(exist_ok=True)
    save_arguments(arguments, arguments.output / 'arguments.json')

    paths1 = {Path(p).stem: Path(p) for p in glob.glob(arguments.input_glob1)}
    paths2 = {Path(p).stem: Path(p) for p in glob.glob(arguments.input_glob2)}
    fn_both_list = set(paths1.keys()) & set(paths2.keys())

    pool = multiprocessing.Pool()
    it = pool.imap(generate_align_indexes, ((paths1[fn], paths2[fn]) for fn in fn_both_list))
    list(tqdm.tqdm(it, total=len(fn_both_list)))


if __name__ == '__main__':
    main()
