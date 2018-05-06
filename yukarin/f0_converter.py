from pathlib import Path
from typing import Dict
from typing import NamedTuple

import numpy

from yukarin.acoustic_feature import AcousticFeature


class Statistics(NamedTuple):
    mean: float
    var: float


class F0Converter(object):
    def __init__(self, input_statistics: Path, target_statistics: Path) -> None:
        d_in: Dict = numpy.load(input_statistics).item()
        self.input_statistics = Statistics(mean=d_in['mean'], var=d_in['var'])

        d_tar: Dict = numpy.load(target_statistics).item()
        self.target_statistics = Statistics(mean=d_tar['mean'], var=d_tar['var'])

    def convert(self, in_feature: AcousticFeature):
        im, iv = self.input_statistics.mean, self.input_statistics.var
        tm, tv = self.target_statistics.mean, self.target_statistics.var

        f0 = numpy.copy(in_feature.f0)
        f0[f0.nonzero()] = numpy.exp((tv / iv) * (numpy.log(f0[f0.nonzero()]) - im) + tm)
        return AcousticFeature(f0=f0)
