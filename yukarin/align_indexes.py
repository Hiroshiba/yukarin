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

    def get_aligned_feature1(self):
        return self.feature1.indexing(self.indexes1)

    def get_aligned_feature2(self):
        return self.feature2.indexing(self.indexes2)

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
        return align_indexes

    def save(
            self,
            path: Path,
            ignores: Tuple[str, ...] = ('feature1', 'feature2'),
    ):
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
    def load(path: Path):
        d = numpy.load(path, allow_pickle=True).item()  # type: dict
        feature = AlignIndexes(
            feature1=d['feature1'],
            feature2=d['feature2'],
            indexes1=d['indexes1'],
            indexes2=d['indexes2'],
        )
        return feature
