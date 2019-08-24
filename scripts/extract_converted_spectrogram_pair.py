"""
before this, should extract converted spectrogram with `convert_acoustic_feature.py`
"""

import argparse
import glob
import multiprocessing
from pathlib import Path
from pprint import pprint
from typing import Tuple

import numpy
import tqdm

from yukarin import AcousticFeature
from yukarin import AlignIndexes
from yukarin.utility.json_utility import save_arguments

parser = argparse.ArgumentParser()
parser.add_argument('--input_glob', '-i', type=Path)
parser.add_argument('--target_glob', '-t', type=Path)
parser.add_argument('--aligned_index_glob', '-ai', type=Path)
parser.add_argument('--output', '-o', type=Path)
parser.add_argument('--enable_overwrite', action='store_true')
arguments = parser.parse_args()

input_glob: Path = arguments.input_glob
target_glob: Path = arguments.target_glob
aligned_index_glob: Path = arguments.aligned_index_glob
output: Path = arguments.output
enable_overwrite: bool = arguments.enable_overwrite


def generate_file(pair_path: Tuple[Path, Path, Path]):
    path1, path2, path_index = pair_path
    if path1.stem != path2.stem:
        print('warning: the file names are different', path1, path2)

    out = Path(output, path1.stem + '.npy')
    if out.exists() and not enable_overwrite:
        return

    indexes = AlignIndexes.load(path_index)
    low_spectrogram = AcousticFeature.load(path=path1).indexing(indexes.indexes1).sp
    high_spectrogram = AcousticFeature.load(path=path2).indexing(indexes.indexes2).sp

    # save
    numpy.save(out.absolute(), {
        'low': low_spectrogram,
        'high': high_spectrogram,
    })


def main():
    pprint(vars(arguments))

    output.mkdir(exist_ok=True)
    save_arguments(arguments, output / 'arguments.json')

    paths1 = {Path(p).stem: Path(p) for p in glob.glob(str(input_glob))}
    paths2 = {Path(p).stem: Path(p) for p in glob.glob(str(target_glob))}
    paths_index = {Path(p).stem: Path(p) for p in glob.glob(str(aligned_index_glob))}
    fn_both_list = set(paths1.keys()) & set(paths2.keys()) & set(paths_index.keys())

    pool = multiprocessing.Pool()
    it = pool.imap(generate_file, ((paths1[fn], paths2[fn], paths_index[fn]) for fn in fn_both_list))
    list(tqdm.tqdm(it, total=len(fn_both_list)))


if __name__ == '__main__':
    main()
