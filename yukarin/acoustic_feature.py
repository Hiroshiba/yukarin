from pathlib import Path
from typing import Dict, List, Iterable

import numpy
import pysptk
import pyworld

from .wave import Wave

_min_mc = -18.3

_is_target = lambda a: isinstance(a, numpy.ndarray)  # numpy.nan is not target


class AcousticFeature(object):
    all_keys = ('f0', 'sp', 'ap', 'coded_ap', 'mc', 'voiced')

    def __init__(
            self,
            f0: numpy.ndarray = numpy.nan,
            sp: numpy.ndarray = numpy.nan,
            ap: numpy.ndarray = numpy.nan,
            coded_ap: numpy.ndarray = numpy.nan,
            mc: numpy.ndarray = numpy.nan,
            mc_wop: numpy.ndarray = numpy.nan,
            voiced: numpy.ndarray = numpy.nan,
    ) -> None:
        assert mc is numpy.nan or mc_wop is numpy.nan

        self.f0 = f0
        self.sp = sp
        self.ap = ap
        self.coded_ap = coded_ap
        self.mc = mc
        if mc_wop is not numpy.nan: self.mc_wop = mc_wop  # block overwrite
        self.voiced = voiced

    @property
    def mc_wop(self):
        pass

    @mc_wop.getter
    def mc_wop(self) -> numpy.ndarray:
        return self.mc[:, 1:] if self.mc is not numpy.nan else numpy.nan

    @mc_wop.setter
    def mc_wop(self, wop: numpy.ndarray):
        assert self.mc is not numpy.nan or wop is not numpy.nan
        if self.mc is numpy.nan:
            self.mc = numpy.ones((wop.shape[0], wop.shape[1] + 1)) * numpy.nan
        self.mc[:, 1:] = wop

    @staticmethod
    def dtypes():
        return dict(
            f0=numpy.float64,
            sp=numpy.float64,
            ap=numpy.float64,
            coded_ap=numpy.float64,
            mc=numpy.float64,
            voiced=numpy.bool,
        )

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
        return AcousticFeature(
            f0=self.f0.astype(dtype) if _is_target(self.f0) else numpy.nan,
            sp=self.sp.astype(dtype) if _is_target(self.sp) else numpy.nan,
            ap=self.ap.astype(dtype) if _is_target(self.ap) else numpy.nan,
            coded_ap=self.coded_ap.astype(dtype) if _is_target(self.coded_ap) else numpy.nan,
            mc=self.mc.astype(dtype) if _is_target(self.mc) else numpy.nan,
            voiced=self.voiced,
        )

    def indexing(self, index: numpy.ndarray):
        return AcousticFeature(
            f0=self.f0[index] if _is_target(self.f0) else numpy.nan,
            sp=self.sp[index] if _is_target(self.sp) else numpy.nan,
            ap=self.ap[index] if _is_target(self.ap) else numpy.nan,
            coded_ap=self.coded_ap[index] if _is_target(self.coded_ap) else numpy.nan,
            mc=self.mc[index] if _is_target(self.mc) else numpy.nan,
            voiced=self.voiced[index] if _is_target(self.voiced) else numpy.nan,
        )

    def indexing_set(self, index: numpy.ndarray, feature: 'AcousticFeature'):
        if _is_target(self.f0):
            self.f0[index] = feature.f0
        if _is_target(self.sp):
            self.sp[index] = feature.sp
        if _is_target(self.ap):
            self.ap[index] = feature.ap
        if _is_target(self.coded_ap):
            self.coded_ap[index] = feature.coded_ap
        if _is_target(self.mc):
            self.mc[index] = feature.mc
        if _is_target(self.voiced):
            self.voiced[index] = feature.voiced

    @staticmethod
    def get_sizes(sampling_rate: int, order: int):
        fft_size = pyworld.get_cheaptrick_fft_size(fs=sampling_rate)
        return dict(
            f0=1,
            sp=fft_size // 2 + 1,
            ap=fft_size // 2 + 1,
            mc=order + 1,
            mc_wop=order,
            voiced=1,
        )

    @classmethod
    def extract_f0(cls, x: numpy.ndarray, fs: int, frame_period: int, f0_floor: float, f0_ceil: float):
        f0, t = pyworld.harvest(
            x,
            fs,
            frame_period=frame_period,
            f0_floor=f0_floor,
            f0_ceil=f0_ceil,
        )
        f0 = pyworld.stonemask(x, f0, t, fs)
        return f0, t

    @classmethod
    def extract(cls, wave: Wave, frame_period, f0_floor, f0_ceil, fft_length, order, alpha, dtype):
        x = wave.wave.astype(numpy.float64)
        fs = wave.sampling_rate

        f0, t = cls.extract_f0(x=x, fs=fs, frame_period=frame_period, f0_floor=f0_floor, f0_ceil=f0_ceil)
        sp = pyworld.cheaptrick(x, f0, t, fs, fft_size=fft_length)
        ap = pyworld.d4c(x, f0, t, fs, fft_size=fft_length)

        mc = pysptk.sp2mc(sp, order=order, alpha=alpha)
        coded_ap = pyworld.code_aperiodicity(ap, fs)
        voiced: numpy.ndarray = ~(f0 == 0)

        if len(x) % fft_length > 0:
            f0 = f0[:-1]
            t = t[:-1]
            sp = sp[:-1]
            ap = ap[:-1]
            mc = mc[:-1]
            coded_ap = coded_ap[:-1]
            voiced = voiced[:-1]

        feature = AcousticFeature(
            f0=f0[:, None],
            sp=sp,
            ap=ap,
            coded_ap=coded_ap,
            mc=mc,
            voiced=voiced[:, None],
        )
        feature = feature.astype_only_float(dtype)
        return feature

    @staticmethod
    def silent(length: int, sizes: Dict[str, int], keys: Iterable[str] = all_keys):
        d = {}
        if 'f0' in keys:
            d['f0'] = numpy.zeros((length, sizes['f0']), dtype=AcousticFeature.dtypes()['f0'])
        if 'sp' in keys:
            d['sp'] = numpy.zeros((length, sizes['sp']), dtype=AcousticFeature.dtypes()['sp'])
        if 'ap' in keys:
            d['ap'] = numpy.zeros((length, sizes['ap']), dtype=AcousticFeature.dtypes()['ap'])
        if 'coded_ap' in keys:
            d['coded_ap'] = numpy.zeros((length, sizes['coded_ap']), dtype=AcousticFeature.dtypes()['coded_ap'])
        if 'mc' in keys:
            d['mc'] = numpy.hstack((
                numpy.ones((length, 1), dtype=AcousticFeature.dtypes()['mc']) * _min_mc,
                numpy.zeros((length, sizes['mc'] - 1), dtype=AcousticFeature.dtypes()['mc'])
            ))
        if 'voiced' in keys:
            d['voiced'] = numpy.zeros((length, sizes['voiced']), dtype=AcousticFeature.dtypes()['voiced'])
        feature = AcousticFeature(**d)
        return feature

    def decode(self, sampling_rate: int, frame_period: float):
        acoustic_feature = self.astype_only_float(numpy.float64)
        out = pyworld.synthesize(
            f0=acoustic_feature.f0.ravel(),
            spectrogram=acoustic_feature.sp,
            aperiodicity=acoustic_feature.ap,
            fs=sampling_rate,
            frame_period=frame_period,
        )
        return Wave(out, sampling_rate=sampling_rate)

    @staticmethod
    def concatenate(fs: List['AcousticFeature'], keys: Iterable[str] = all_keys):
        return AcousticFeature(**{
            key: numpy.concatenate([getattr(f, key) for f in fs]) if _is_target(getattr(fs[0], key)) else numpy.nan
            for key in keys
        })

    def pick(self, first: int, last: int, keys: Iterable[str] = all_keys):
        return AcousticFeature(**{
            key: getattr(self, key)[first:last] if _is_target(getattr(self, key)) else numpy.nan
            for key in keys
        })

    def save(self, path: Path, ignores: Iterable[str] = None):
        keys = filter(lambda k: k not in ignores, self.all_keys)
        d = {k: getattr(self, k) for k in keys}
        numpy.save(path, d)

    @staticmethod
    def load(path: Path):
        d: Dict = numpy.load(path, allow_pickle=True).item()
        return AcousticFeature(**d)

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
