import argparse
import glob
import multiprocessing
import re
from functools import partial
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy
from become_yukarin import SuperResolution
from become_yukarin.config.sr_config import create_from_json as create_sr_config
from become_yukarin.data_struct import AcousticFeature as BYAcousticFeature

from yukarin import AcousticConverter
from yukarin.config import create_from_json as create_config
from yukarin.f0_converter import F0Converter
from yukarin.utility.json_utility import save_arguments

parser = argparse.ArgumentParser()
parser.add_argument('--voice_changer_model_dir', '-vcmd', type=Path)
parser.add_argument('--voice_changer_model_iteration', '-vcmi', type=int)
parser.add_argument('--voice_changer_config', '-vcc', type=Path)
parser.add_argument('--input_wave_scale', '-iws', type=float)
parser.add_argument('--out_sampling_rate', '-osr', type=int)
parser.add_argument('--filter_size', '-fs', type=int)
parser.add_argument('--super_resolution_model', '-srm', type=Path)
parser.add_argument('--super_resolution_config', '-src', type=Path)
parser.add_argument('--input_statistics', '-is', type=Path)
parser.add_argument('--target_statistics', '-ts', type=Path)
parser.add_argument('--output_dir', '-o', type=Path, default='./output/')
parser.add_argument('--dataset_input_wave_dir', '-diwd', type=Path)
parser.add_argument('--dataset_target_wave_dir', '-dtwd', type=Path)
parser.add_argument('--test_wave_dir', '-twd', type=Path)
parser.add_argument('--gpu', type=int)
arguments = parser.parse_args()

voice_changer_model_dir: Path = arguments.voice_changer_model_dir
voice_changer_model_iteration: int = arguments.voice_changer_model_iteration
voice_changer_config: Path = arguments.voice_changer_config
filter_size: int = arguments.filter_size
super_resolution_model: Path = arguments.super_resolution_model
super_resolution_config: Path = arguments.super_resolution_config
input_statistics: Path = arguments.input_statistics
target_statistics: Path = arguments.target_statistics
output_dir: Path = arguments.output_dir
dataset_input_wave_dir: Path = arguments.dataset_input_wave_dir
dataset_target_wave_dir: Path = arguments.dataset_target_wave_dir
test_wave_dir: Path = arguments.test_wave_dir
gpu: int = arguments.gpu

output_dir.mkdir(exist_ok=True)

output = output_dir / voice_changer_model_dir.name
output.mkdir(exist_ok=True)


def _extract_number(f):
    s = re.findall("\d+", str(f))
    return int(s[-1]) if s else -1


def process(p_in: Path, acoustic_converter: AcousticConverter, super_resolution: SuperResolution):
    if p_in.suffix in ['.npy', '.npz']:
        p_in = Path(glob.glob(str(dataset_input_wave_dir / p_in.stem) + '.*')[0])

    w_in = acoustic_converter.load_wave(p_in)
    w_in.wave *= arguments.input_wave_scale
    f_in = acoustic_converter.extract_acoustic_feature(w_in)
    f_in_effective, effective = acoustic_converter.separate_effective(wave=w_in, feature=f_in)
    f_low = acoustic_converter.convert(f_in_effective)
    f_low = acoustic_converter.combine_silent(effective=effective, feature=f_low)
    if filter_size is not None:
        f_low.f0 = AcousticConverter.filter_f0(f_low.f0, filter_size=filter_size)
    f_low = acoustic_converter.decode_spectrogram(f_low)
    s_high = super_resolution.convert(f_low.sp.astype(numpy.float32))

    # target
    paths = glob.glob(str(dataset_target_wave_dir / p_in.stem) + '.*')
    has_true = len(paths) > 0
    if has_true:
        p_true = Path(paths[0])
        w_true = acoustic_converter.load_wave(p_true)
        f_true = acoustic_converter.extract_acoustic_feature(w_true)

    # save figure
    fig = plt.figure(figsize=[18, 8])

    plt.subplot(4, 1, 1)
    plt.imshow(numpy.log(f_in.sp).T, aspect='auto', origin='reverse')
    plt.plot(f_in.f0, 'w')
    plt.colorbar()

    plt.subplot(4, 1, 2)
    plt.imshow(numpy.log(f_low.sp).T, aspect='auto', origin='reverse')
    plt.plot(f_low.f0, 'w')
    plt.colorbar()

    plt.subplot(4, 1, 3)
    plt.imshow(numpy.log(s_high).T, aspect='auto', origin='reverse')
    plt.colorbar()

    if has_true:
        plt.subplot(4, 1, 4)
        plt.imshow(numpy.log(f_true.sp).T, aspect='auto', origin='reverse')
        plt.plot(f_true.f0, 'w')
        plt.colorbar()

    fig.savefig(output / (p_in.stem + '.png'))

    # save wave
    f_low_sr = BYAcousticFeature(
        f0=f_low.f0,
        spectrogram=f_low.sp,
        aperiodicity=f_low.ap,
        mfcc=f_low.mc,
        voiced=f_low.voiced,
    )

    rate = acoustic_converter.out_sampling_rate
    wave = super_resolution(s_high, acoustic_feature=f_low_sr, sampling_rate=rate)
    librosa.output.write_wav(y=wave.wave, path=str(output / (p_in.stem + '.wav')), sr=rate)


def main():
    save_arguments(arguments, output / 'arguments.json')

    if arguments.voice_changer_model_iteration is None:
        paths = voice_changer_model_dir.glob('predictor_*.npz')
        voice_changer_model = list(sorted(paths, key=_extract_number))[-1]
    else:
        voice_changer_model = voice_changer_model_dir / 'predictor_{}.npz'.format(
            arguments.voice_changer_model_iteration)

    # f0 converter
    if input_statistics is not None:
        f0_converter = F0Converter(input_statistics=input_statistics, target_statistics=target_statistics)
    else:
        f0_converter = None

    # acoustic converter
    config = create_config(arguments.voice_changer_config)
    acoustic_converter = AcousticConverter(
        config,
        voice_changer_model,
        gpu=arguments.gpu,
        f0_converter=f0_converter,
        out_sampling_rate=arguments.out_sampling_rate,
    )

    # super resolution
    sr_config = create_sr_config(arguments.super_resolution_config)
    super_resolution = SuperResolution(sr_config, super_resolution_model, gpu=arguments.gpu)

    # dataset's test
    input_paths = list(sorted([Path(p) for p in glob.glob(str(config.dataset.input_glob))]))
    numpy.random.RandomState(config.dataset.seed).shuffle(input_paths)
    paths_test = input_paths[-config.dataset.num_test:]

    # test data
    if test_wave_dir is not None:
        paths_test += list(test_wave_dir.glob('*.wav'))

    process_partial = partial(process, acoustic_converter=acoustic_converter, super_resolution=super_resolution)
    if gpu is None:
        pool = multiprocessing.Pool()
        pool.map(process_partial, paths_test)
    else:
        list(map(process_partial, paths_test))


if __name__ == '__main__':
    main()
