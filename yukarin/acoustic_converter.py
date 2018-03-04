from functools import partial
from pathlib import Path

import chainer
import numpy
import pysptk
import pyworld

from yukarin.acoustic_feature import AcousticFeature
from yukarin.config import Config
from yukarin.dataset import decode_feature
from yukarin.dataset import encode_feature
from yukarin.model import create_predictor
from yukarin.wave import Wave


class AcousticConverter(object):
    def __init__(self, config: Config, model_path: Path, gpu: int = None, out_sampling_rate: int = None) -> None:
        if out_sampling_rate is None:
            out_sampling_rate = config.dataset.acoustic_param.sampling_rate

        self.config = config
        self.model_path = model_path
        self.gpu = gpu
        self.out_sampling_rate = out_sampling_rate
        self._param = self.config.dataset.acoustic_param

        self.model = model = create_predictor(config.model)
        chainer.serializers.load_npz(str(model_path), model)
        if self.gpu is not None:
            model.to_gpu(self.gpu)

    def _encode_feature(self, data):
        return encode_feature(data, targets=self.config.dataset.features)

    def _decode_feature(self, data):
        sizes = AcousticFeature.get_sizes(
            sampling_rate=self._param.sampling_rate,
            order=self._param.order,
        )
        return decode_feature(data, targets=self.config.dataset.features, sizes=sizes)

    def load_wave(self, path: Path):
        return Wave.load(path, sampling_rate=self._param.sampling_rate)

    def extract_acoustic_feature(self, wave: Wave):
        return AcousticFeature.extract(
            wave,
            frame_period=self._param.frame_period,
            f0_floor=self._param.f0_floor,
            f0_ceil=self._param.f0_ceil,
            fft_length=self._param.fft_length,
            order=self._param.order,
            alpha=self._param.alpha,
            dtype=self._param.dtype,
        )

    def load_acoustic_feature(self, path: Path):
        return AcousticFeature.load(path)

    def convert(self, in_feature: AcousticFeature):
        input = self._encode_feature(in_feature)

        pad = 128 - input.shape[1] % 128
        input = numpy.pad(input, [(0, 0), (0, pad)], mode='minimum')

        converter = partial(chainer.dataset.convert.concat_examples, device=self.gpu, padding=0)
        inputs = converter([input])

        with chainer.using_config('train', False):
            out = self.model(inputs).data[0]

        if self.gpu is not None:
            out = chainer.cuda.to_cpu(out)
        out = out[:, :-pad]

        out = self._decode_feature(out)
        out.ap = in_feature.ap
        out.voiced = in_feature.voiced
        out.f0[~out.voiced] = 0

        fftlen = pyworld.get_cheaptrick_fft_size(self.out_sampling_rate)
        sp = pysptk.mc2sp(
            out.mc,
            alpha=self._param.alpha,
            fftlen=fftlen,
        )
        out.sp = sp

        out = out.astype_only_float(numpy.float64)
        return out

    def decode_acoustic_feature(self, feature: AcousticFeature):
        out = pyworld.synthesize(
            f0=feature.f0.ravel(),
            spectrogram=feature.sp,
            aperiodicity=feature.ap,
            fs=self.out_sampling_rate,
            frame_period=self._param.frame_period,
        )
        return Wave(out, sampling_rate=self.out_sampling_rate)
