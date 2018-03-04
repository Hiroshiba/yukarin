from pathlib import Path
from typing import Tuple

import numpy
import pysptk
import pyworld

from .wave import Wave


class AcousticFeature(object):
    def __init__(
            self,
            f0: numpy.ndarray = numpy.nan,
            sp: numpy.ndarray = numpy.nan,
            ap: numpy.ndarray = numpy.nan,
            coded_ap: numpy.ndarray = numpy.nan,
            mc: numpy.ndarray = numpy.nan,
            voiced: numpy.ndarray = numpy.nan,
    ) -> None:
        self.f0 = f0
        self.sp = sp
        self.ap = ap
        self.coded_ap = coded_ap
        self.mc = mc
        self.voiced = voiced

    def astype(self, dtype):
        return AcousticFeature(
            f0=self.f0.astype(dtype),
            sp=self.sp.astype(dtype),
            ap=self.ap.astype(dtype),
            coded_ap=self.coded_ap.astype(dtype),
            mc=self.mc.astype(dtype),
            voiced=self.voiced.astype(dtype),
        )

    def astype_only_float(self, dtype):
        is_target = lambda a: not numpy.any(numpy.isnan(a))
        return AcousticFeature(
            f0=self.f0.astype(dtype) if is_target(self.f0) else numpy.nan,
            sp=self.sp.astype(dtype) if is_target(self.sp) else numpy.nan,
            ap=self.ap.astype(dtype) if is_target(self.ap) else numpy.nan,
            coded_ap=self.coded_ap.astype(dtype) if is_target(self.coded_ap) else numpy.nan,
            mc=self.mc.astype(dtype) if is_target(self.mc) else numpy.nan,
            voiced=self.voiced,
        )

    def indexing(self, index: numpy.ndarray):
        is_target = lambda a: not numpy.any(numpy.isnan(a))
        return AcousticFeature(
            f0=self.f0[index] if is_target(self.f0) else numpy.nan,
            sp=self.sp[index] if is_target(self.sp) else numpy.nan,
            ap=self.ap[index] if is_target(self.ap) else numpy.nan,
            coded_ap=self.coded_ap[index] if is_target(self.coded_ap) else numpy.nan,
            mc=self.mc[index] if is_target(self.mc) else numpy.nan,
            voiced=self.voiced[index] if is_target(self.voiced) else numpy.nan,
        )

    def validate(self):
        assert self.f0.ndim == 2
        assert self.sp.ndim == 2
        assert self.ap.ndim == 2
        assert self.coded_ap.ndim == 2
        assert self.mc.ndim == 2
        assert self.voiced.ndim == 2

        len_time = len(self.f0)
        assert len(self.sp) == len_time
        assert len(self.ap) == len_time
        assert len(self.coded_ap) == len_time
        assert len(self.mc) == len_time
        assert len(self.voiced) == len_time

        assert self.voiced.dtype == numpy.bool

    @staticmethod
    def get_sizes(sampling_rate: int, order: int):
        fft_size = pyworld.get_cheaptrick_fft_size(fs=sampling_rate)
        return dict(
            f0=1,
            sp=fft_size // 2 + 1,
            ap=fft_size // 2 + 1,
            mc=order + 1,
            voiced=1,
        )

    @staticmethod
    def extract(wave: Wave, frame_period, f0_floor, f0_ceil, fft_length, order, alpha, dtype):
        x = wave.wave.astype(numpy.float64)
        fs = wave.sampling_rate

        f0, t = pyworld.harvest(
            x,
            fs,
            frame_period=frame_period,
            f0_floor=f0_floor,
            f0_ceil=f0_ceil,
        )

        f0 = pyworld.stonemask(x, f0, t, fs)
        sp = pyworld.cheaptrick(x, f0, t, fs, fft_size=fft_length)
        ap = pyworld.d4c(x, f0, t, fs, fft_size=fft_length)

        mc = pysptk.sp2mc(sp, order=order, alpha=alpha)
        coded_ap = pyworld.code_aperiodicity(ap, fs)
        voiced = ~(f0 == 0)  # type: numpy.ndarray

        feature = AcousticFeature(
            f0=f0[:, None],
            sp=sp,
            ap=ap,
            coded_ap=coded_ap,
            mc=mc,
            voiced=voiced[:, None],
        )
        feature = feature.astype_only_float(dtype)
        feature.validate()
        return feature

    def decode(self, sampling_rate: int, frame_period: float):
        acoustic_feature = self.astype_only_float(numpy.float64)
        out = pyworld.synthesize(
            f0=acoustic_feature.f0.ravel(),
            spectrogram=acoustic_feature.sp,
            aperiodicity=acoustic_feature.ap,
            fs=sampling_rate,
            frame_period=frame_period
        )
        return Wave(out, sampling_rate=sampling_rate)

    def save(self, path: Path, validate=False, ignores: Tuple[str] = None):
        if validate:
            self.validate()

        d = dict(
            f0=self.f0,
            sp=self.sp,
            ap=self.ap,
            coded_ap=self.coded_ap,
            mc=self.mc,
            voiced=self.voiced,
        )
        for k in ignores:
            assert k in d
            d[k] = numpy.nan

        numpy.save(path, d)

    @staticmethod
    def load(path: Path, validate=False):
        d = numpy.load(path).item()  # type: dict
        feature = AcousticFeature(
            f0=d['f0'],
            sp=d['sp'],
            ap=d['ap'],
            coded_ap=d['coded_ap'],
            mc=d['mc'],
            voiced=d['voiced'],
        )
        if validate:
            feature.validate()
        return feature

    @staticmethod
    def mc2sp(mc: numpy.ndarray, sampling_rate: int, alpha: float):
        return pysptk.mc2sp(
            mc.astype(numpy.float64),
            alpha=alpha,
            fftlen=pyworld.get_cheaptrick_fft_size(sampling_rate),
        )

    @staticmethod
    def decode_ap(ap: numpy.ndarray, sampling_rate: int):
        return pyworld.decode_aperiodicity(
            ap.astype(numpy.float64),
            sampling_rate,
            pyworld.get_cheaptrick_fft_size(sampling_rate),
        )
