from functools import partial
from pathlib import Path

import chainer
import librosa
import numpy
import pysptk
import pyworld

from yukarin.acoustic_feature import AcousticFeature
from yukarin.config import Config
from yukarin.dataset import decode_feature
from yukarin.dataset import encode_feature
from yukarin.f0_converter import F0Converter
from yukarin.model import create_predictor
from yukarin.wave import Wave


class AcousticConverter(object):
    def __init__(
            self,
            config: Config,
            model_path: Path,
            gpu: int = None,
            f0_converter: F0Converter = None,
            out_sampling_rate: int = None,
    ) -> None:
        if out_sampling_rate is None:
            out_sampling_rate = config.dataset.acoustic_param.sampling_rate

        self.config = config
        self.model_path = model_path
        self.gpu = gpu
        self.f0_converter = f0_converter
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

    def separate_effective(self, wave: Wave, feature: AcousticFeature, threshold=None):
        """
        :return: (effective feature, effective flags)
        """
        hop, length = wave.get_hop_and_length(frame_period=self._param.frame_period)
        if threshold is None:
            if self._param.threshold_db is not None:
                effective = wave.get_effective_frame(
                    threshold_db=self._param.threshold_db,
                    fft_length=self._param.fft_length,
                    frame_period=self._param.frame_period,
                )
                feature = feature.indexing(effective)
            else:
                effective = numpy.ones(length, dtype=bool)
        else:
            mse = librosa.feature.rmse(y=wave.wave, frame_length=self._param.fft_length, hop_length=hop) ** 2
            effective = (librosa.core.power_to_db(mse.squeeze()) > - threshold)
            if len(effective) < len(feature.f0):  # the divide move
                effective = numpy.r_[effective, False]
            if len(effective) > len(feature.f0):  # the divide move
                effective = effective
            feature = feature.indexing(effective)
        return feature, effective

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

        if numpy.any(numpy.isnan(out.f0)):
            if self.f0_converter is not None:
                out.f0 = self.f0_converter.convert(in_feature.f0)
            else:
                out.f0 = in_feature.f0

        out.f0[~out.voiced] = 0
        return out

    @staticmethod
    def filter_f0(f0: numpy.ndarray, filter_size: int):
        import scipy.ndimage
        return scipy.ndimage.median_filter(f0, size=(filter_size, 1))

    def combine_silent(self, effective: numpy.ndarray, feature: AcousticFeature):
        sizes = AcousticFeature.get_sizes(
            sampling_rate=self._param.sampling_rate,
            order=self._param.order,
        )
        silent_feature = AcousticFeature.silent(len(effective), sizes=sizes, keys=('mc', 'ap', 'f0', 'voiced'))
        silent_feature.indexing_set(effective, feature)
        return silent_feature

    def decode_spectrogram(self, feature: AcousticFeature):
        fftlen = pyworld.get_cheaptrick_fft_size(self.out_sampling_rate)
        feature.sp = pysptk.mc2sp(
            feature.mc.astype(numpy.float32),
            alpha=pysptk.util.mcepalpha(self.out_sampling_rate),
            fftlen=fftlen,
        )
        return feature

    def decode_acoustic_feature(self, feature: AcousticFeature):
        out = pyworld.synthesize(
            f0=feature.f0.ravel(),
            spectrogram=feature.sp,
            aperiodicity=feature.ap,
            fs=self.out_sampling_rate,
            frame_period=self._param.frame_period,
        )
        return Wave(out, sampling_rate=self.out_sampling_rate)
