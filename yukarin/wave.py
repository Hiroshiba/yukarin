from pathlib import Path

import librosa
import numpy


class Wave(object):
    def __init__(
            self,
            wave: numpy.ndarray,
            sampling_rate: int,
    ) -> None:
        self.wave = wave
        self.sampling_rate = sampling_rate

    @staticmethod
    def load(path: Path, sampling_rate: int, dtype=numpy.float32):
        wave = librosa.core.load(str(path), sr=sampling_rate, dtype=dtype)[0]
        return Wave(wave=wave, sampling_rate=sampling_rate)

    def pad(self, pre_second: int, post_second: int):
        pre, post = int(self.sampling_rate * pre_second), int(self.sampling_rate * post_second)
        return Wave(
            wave=numpy.pad(self.wave, pad_width=(pre, post), mode='constant'),
            sampling_rate=self.sampling_rate,
        )

    def get_hop_and_length(self, frame_period: float):
        hop = round(self.sampling_rate * frame_period // 1000)
        length = int(len(self.wave) / hop) + 1
        return hop, length

    def get_effective_frame(self, threshold_db: float, fft_length: int, frame_period: float):
        hop, length = self.get_hop_and_length(frame_period)
        effective = numpy.zeros(length, dtype=bool)

        s = librosa.effects.split(y=self.wave, top_db=threshold_db, frame_length=fft_length, hop_length=hop) // hop
        for a, b in s:
            effective[a:b] = True
        return effective
