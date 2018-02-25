from pathlib import Path
from typing import Tuple

import numpy
from become_yukarin.dataset.utility import MelCepstrumAligner

from .acoustic_feature import AcousticFeature


class AlignIndexes(object):
    def __init__(
            self,
            feature1: AcousticFeature,
            feature2: AcousticFeature,
            indexes1: numpy.ndarray,
            indexes2: numpy.ndarray,
    ) -> None:
        self.feature1 = feature1
        self.feature2 = feature2
        self.indexes1 = indexes1
        self.indexes2 = indexes2

    def validate(self):
        assert numpy.all(self.indexes1 == numpy.int64(self.indexes1))
        assert numpy.all(self.indexes2 == numpy.int64(self.indexes2))
        assert len(self.indexes1) == len(self.indexes2)

    @staticmethod
    def get_aligned_feature(feature: AcousticFeature, indexes: numpy.ndarray):
        is_target = lambda a: not numpy.any(numpy.isnan(a))
        return AcousticFeature(
            f0=feature.f0[indexes] if is_target(feature.f0) else numpy.nan,
            sp=feature.sp[indexes] if is_target(feature.sp) else numpy.nan,
            ap=feature.ap[indexes] if is_target(feature.ap) else numpy.nan,
            coded_ap=feature.coded_ap[indexes] if is_target(feature.coded_ap) else numpy.nan,
            mc=feature.mc[indexes] if is_target(feature.mc) else numpy.nan,
            voiced=feature.voiced[indexes] if is_target(feature.voiced) else numpy.nan,
        )

    def get_aligned_feature1(self):
        return self.get_aligned_feature(self.feature1, self.indexes1)

    def get_aligned_feature2(self):
        return self.get_aligned_feature(self.feature2, self.indexes2)

    @staticmethod
    def extract(feature1: AcousticFeature, feature2: AcousticFeature, dtype='int64'):
        aligner = MelCepstrumAligner(feature1.mc, feature2.mc)
        indexes1 = (aligner.normed_path_x * len(feature1.mc)).astype(dtype)
        indexes2 = (aligner.normed_path_y * len(feature2.mc)).astype(dtype)

        align_indexes = AlignIndexes(
            feature1=feature1,
            feature2=feature2,
            indexes1=indexes1,
            indexes2=indexes2,
        )
        align_indexes.validate()
        return align_indexes

    def save(
            self,
            path: Path,
            validate=False,
            ignores: Tuple[str] = ('feature1', 'feature2'),
    ):
        if validate:
            self.validate()

        d = dict(
            feature1=self.feature1,
            feature2=self.feature2,
            indexes1=self.indexes1,
            indexes2=self.indexes2,
        )
        for k in ignores:
            assert k in d
            d[k] = numpy.nan

        numpy.save(path, d)

    @staticmethod
    def load(path: Path, validate=False):
        d = numpy.load(path).item()  # type: dict
        feature = AlignIndexes(
            feature1=d['feature1'],
            feature2=d['feature2'],
            indexes1=d['indexes1'],
            indexes2=d['indexes2'],
        )
        if validate:
            feature.validate()
        return feature
