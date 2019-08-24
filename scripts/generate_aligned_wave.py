"""
extract indexes for alignment.
"""

import argparse
import glob
import multiprocessing
from functools import partial
from pathlib import Path
from pprint import pprint
from typing import Tuple

import librosa
import numpy
import tqdm

from yukarin.acoustic_feature import AcousticFeature
from yukarin.align_indexes import AlignIndexes
from yukarin.param import AcousticParam
from yukarin.utility.json_utility import save_arguments

base_acoustic_param = AcousticParam()

parser = argparse.ArgumentParser()
parser.add_argument('--input_feature_glob1', '-if1')
parser.add_argument('--input_feature_glob2', '-if2')
parser.add_argument('--input_indexes', '-ii')
parser.add_argument('--output', '-o', type=Path)
parser.add_argument('--sampling_rate', type=int, default=base_acoustic_param.sampling_rate)
parser.add_argument('--frame_period', type=float, default=base_acoustic_param.frame_period)
parser.add_argument('--alpha', type=float, default=base_acoustic_param.alpha)
parser.add_argument('--disable_overwrite', action='store_true')
arguments = parser.parse_args()


def generate_aligned_wave(
        pair_path: Tuple[Path, Path, Path],
        sampling_rate: int,
        frame_period: float,
        alpha: float,
):
    path_feature1, path_feature2, path_indexes = pair_path

    if path_feature1.stem != path_feature2.stem:
        print('warning: the file names are different', path_feature1, path_feature2)

    if path_feature1.stem != path_indexes.stem:
        print('warning: the file names are different', path_feature1, path_indexes)

    out = Path(arguments.output, path_indexes.stem + '.wav')
    if arguments.disable_overwrite:
        return

    feature1 = AcousticFeature.load(path=path_feature1)
    feature2 = AcousticFeature.load(path=path_feature2)
    feature1.sp = AcousticFeature.mc2sp(feature1.mc, sampling_rate=sampling_rate, alpha=alpha)
    feature2.sp = AcousticFeature.mc2sp(feature2.mc, sampling_rate=sampling_rate, alpha=alpha)
    feature1.ap = AcousticFeature.decode_ap(feature1.coded_ap, sampling_rate=sampling_rate)
    feature2.ap = AcousticFeature.decode_ap(feature2.coded_ap, sampling_rate=sampling_rate)

    align_indexes = AlignIndexes.load(path=path_indexes)
    align_indexes.feature1 = feature1
    align_indexes.feature2 = feature2

    wave1 = align_indexes.get_aligned_feature1().decode(sampling_rate=sampling_rate, frame_period=frame_period)
    wave2 = align_indexes.get_aligned_feature2().decode(sampling_rate=sampling_rate, frame_period=frame_period)

    # save
    y = numpy.vstack([wave1.wave, wave2.wave])
    librosa.output.write_wav(str(out), y, sr=sampling_rate)


def main():
    pprint(vars(arguments))

    arguments.output.mkdir(exist_ok=True)
    save_arguments(arguments, arguments.output / 'arguments.json')

    path_feature1 = {Path(p).stem: Path(p) for p in glob.glob(arguments.input_feature_glob1)}
    path_feature2 = {Path(p).stem: Path(p) for p in glob.glob(arguments.input_feature_glob2)}

    path_indexes = {Path(p).stem: Path(p) for p in glob.glob(arguments.input_indexes)}
    fn_both_list = set(path_feature1.keys()) & set(path_indexes.keys())

    pool = multiprocessing.Pool()
    generate = partial(
        generate_aligned_wave,
        sampling_rate=arguments.sampling_rate,
        frame_period=arguments.frame_period,
        alpha=arguments.alpha,
    )
    it = pool.imap(generate, ((path_feature1[fn], path_feature2[fn], path_indexes[fn]) for fn in fn_both_list))
    list(tqdm.tqdm(it, total=len(path_feature1)))


if __name__ == '__main__':
    main()
