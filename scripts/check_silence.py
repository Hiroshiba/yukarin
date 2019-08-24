"""
check silence of wave.
"""

import argparse
import glob
import math
import multiprocessing
from pathlib import Path
from pprint import pprint

import matplotlib.pyplot as plt
import numpy
import tqdm
from sprocket.speech import FeatureExtractor

from yukarin import Wave
from yukarin.param import AcousticParam
from yukarin.utility.sprocket_utility import SpeakerYML
from yukarin.utility.sprocket_utility import low_cut_filter

base_acoustic_param = AcousticParam()

parser = argparse.ArgumentParser()
parser.add_argument('--input_wave_glob', '-i')
parser.add_argument('--candidate_threshold', '-th', nargs='+', type=float)
parser.add_argument('--output_image', '-o', type=Path)
parser.add_argument('--speaker_yml', type=Path)
parser.add_argument('--pad_second', type=float, default=base_acoustic_param.pad_second)
arguments = parser.parse_args()

# read parameters from speaker yml
sconf = SpeakerYML(arguments.speaker_yml)

# constract FeatureExtractor class
feat = FeatureExtractor(
    analyzer=sconf.analyzer,
    fs=sconf.wav_fs,
    fftl=sconf.wav_fftl,
    shiftms=sconf.wav_shiftms,
    minf0=sconf.f0_minf0,
    maxf0=sconf.f0_maxf0,
)


def calc_score(path: Path):
    scores = []

    wave = Wave.load(path=path, sampling_rate=sconf.wav_fs)
    wave = wave.pad(pre_second=arguments.pad_second, post_second=arguments.pad_second)

    hop = sconf.wav_fs * sconf.wav_shiftms // 1000
    length = int(math.ceil(len(wave.wave) / hop + 0.0001))

    # for sprocket
    x = low_cut_filter(wave.wave, wave.sampling_rate, cutoff=70)
    feat.analyze(x)
    npow = feat.npow()
    effective1: numpy.ndarray = (npow > sconf.power_threshold)
    assert len(effective1) == length, str(path)

    # for yukarin
    for th in arguments.candidate_threshold:
        effective2 = wave.get_effective_frame(
            threshold_db=th,
            fft_length=sconf.wav_fftl,
            frame_period=sconf.wav_shiftms,
        )
        scores.append([
            (effective1 == effective2).sum(),
            (effective1 == effective2)[effective1].sum(),
            (effective1 == effective2)[~effective1].sum(),
            length,
        ])

    return scores


def main():
    pprint(vars(arguments))

    paths = [Path(p) for p in sorted(glob.glob(arguments.input_wave_glob))]
    pool = multiprocessing.Pool()
    it = pool.imap(calc_score, paths)
    scores_list = list(tqdm.tqdm(it, total=len(paths)))

    pprint({
        th: score
        for th, score in zip(arguments.candidate_threshold, numpy.array(scores_list).sum(axis=0))
    })

    fig = plt.figure(figsize=[10, 5])
    plt.plot(arguments.candidate_threshold, numpy.array(scores_list).sum(axis=0))
    fig.savefig(arguments.output_image)


if __name__ == '__main__':
    main()
