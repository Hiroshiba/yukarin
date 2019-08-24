"""
extract indexes for alignment with GMM.
"""

import argparse
import glob
import multiprocessing
from pathlib import Path
from pprint import pprint
from typing import Tuple

import numpy
import tqdm
from sklearn.externals import joblib
from sprocket.model import GMMConvertor
from sprocket.speech import FeatureExtractor
from sprocket.util import static_delta

from yukarin import Wave
from yukarin.acoustic_feature import AcousticFeature
from yukarin.align_indexes import AlignIndexes
from yukarin.param import AcousticParam
from yukarin.utility.json_utility import save_arguments
from yukarin.utility.sprocket_utility import PairYML
from yukarin.utility.sprocket_utility import SpeakerYML
from yukarin.utility.sprocket_utility import low_cut_filter

base_acoustic_param = AcousticParam()

parser = argparse.ArgumentParser()
parser.add_argument('--input_wave_glob1', '-i1')
parser.add_argument('--input_wave_glob2', '-i2')
parser.add_argument('--output', '-o', type=Path)
parser.add_argument('--org_yml', type=Path)
parser.add_argument('--tar_yml', type=Path)
parser.add_argument('--pair_yml', type=Path)
parser.add_argument('--gmm', type=Path)
parser.add_argument('--pad_second1', type=float, default=base_acoustic_param.pad_second)
parser.add_argument('--pad_second2', type=float, default=base_acoustic_param.pad_second)
parser.add_argument('--threshold_db1', type=float, default=base_acoustic_param.threshold_db)
parser.add_argument('--threshold_db2', type=float, default=base_acoustic_param.threshold_db)
parser.add_argument('--dtype', type=str, default='int64')
parser.add_argument('--ignore_feature', nargs='+', default=('feature1', 'feature2'))
parser.add_argument('--enable_overwrite', action='store_true')
arguments = parser.parse_args()

# read parameters from speaker yml
sconf1 = SpeakerYML(arguments.org_yml)
sconf2 = SpeakerYML(arguments.tar_yml)
pconf = PairYML(arguments.pair_yml)

# read GMM for mcep
mcepgmm = GMMConvertor(
    n_mix=pconf.GMM_mcep_n_mix,
    covtype=pconf.GMM_mcep_covtype,
    gmmmode=None,
)
param = joblib.load(arguments.gmm)
mcepgmm.open_from_param(param)

# constract FeatureExtractor class
feat1 = FeatureExtractor(
    analyzer=sconf1.analyzer,
    fs=sconf1.wav_fs,
    fftl=sconf1.wav_fftl,
    shiftms=sconf1.wav_shiftms,
    minf0=sconf1.f0_minf0,
    maxf0=sconf1.f0_maxf0,
)
feat2 = FeatureExtractor(
    analyzer=sconf2.analyzer,
    fs=sconf2.wav_fs,
    fftl=sconf2.wav_fftl,
    shiftms=sconf2.wav_shiftms,
    minf0=sconf2.f0_minf0,
    maxf0=sconf2.f0_maxf0,
)


def generate_align_indexes(pair_path: Tuple[Path, Path]):
    path1, path2 = pair_path
    if path1.stem != path2.stem:
        print('warning: the file names are different', path1, path2)

    out = Path(arguments.output, path1.stem + '.npy')
    if out.exists() and not arguments.enable_overwrite:
        return

    # original
    wave = Wave.load(path=path1, sampling_rate=sconf1.wav_fs)
    wave = wave.pad(pre_second=arguments.pad_second1, post_second=arguments.pad_second1)
    x = low_cut_filter(wave.wave, wave.sampling_rate, cutoff=70)

    feat1.analyze(x)
    mcep = feat1.mcep(dim=sconf1.mcep_dim, alpha=sconf1.mcep_alpha)

    if arguments.threshold_db1 is not None:
        indexes = wave.get_effective_frame(
            threshold_db=arguments.threshold_db1,
            fft_length=sconf1.wav_fftl,
            frame_period=sconf1.wav_shiftms,
        )
        mcep = mcep[indexes]

    cvmcep_wopow = mcepgmm.convert(static_delta(mcep[:, 1:]), cvtype=pconf.GMM_mcep_cvtype)
    mcep1 = numpy.c_[mcep[:, 0], cvmcep_wopow]

    # target
    wave = Wave.load(path=path2, sampling_rate=sconf2.wav_fs)
    wave = wave.pad(pre_second=arguments.pad_second2, post_second=arguments.pad_second2)
    x = low_cut_filter(wave.wave, wave.sampling_rate, cutoff=70)

    feat2.analyze(x)
    mcep2 = feat2.mcep(dim=sconf2.mcep_dim, alpha=sconf2.mcep_alpha)

    if arguments.threshold_db2 is not None:
        indexes = wave.get_effective_frame(
            threshold_db=arguments.threshold_db2,
            fft_length=sconf2.wav_fftl,
            frame_period=sconf2.wav_shiftms,
        )
        mcep2 = mcep2[indexes]

    # align
    feature1 = AcousticFeature(mc=mcep1)
    feature2 = AcousticFeature(mc=mcep2)
    align_indexes = AlignIndexes.extract(feature1, feature2, dtype=arguments.dtype)
    align_indexes.save(path=out, ignores=arguments.ignore_feature)


def main():
    pprint(vars(arguments))

    arguments.output.mkdir(exist_ok=True)
    save_arguments(arguments, arguments.output / 'arguments.json')

    paths1 = [Path(p) for p in sorted(glob.glob(arguments.input_wave_glob1))]
    paths2 = [Path(p) for p in sorted(glob.glob(arguments.input_wave_glob2))]
    assert len(paths1) == len(paths2)

    pool = multiprocessing.Pool()
    it = pool.imap(generate_align_indexes, zip(paths1, paths2))
    list(tqdm.tqdm(it, total=len(paths1)))


if __name__ == '__main__':
    main()
