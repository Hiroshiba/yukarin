class AcousticParam(object):
    def __init__(
            self,
            sampling_rate: int = 24000,
            pad_second: float = 0,
            threshold_db: float = None,
            frame_period: int = 5,
            order: int = 8,
            alpha: float = 0.466,
            f0_floor: float = 71,
            f0_ceil: float = 800,
            fft_length: int = 1024,
            dtype: str = 'float32',
    ) -> None:
        self.sampling_rate = sampling_rate
        self.pad_second = pad_second
        self.threshold_db = threshold_db
        self.frame_period = frame_period
        self.order = order
        self.alpha = alpha
        self.f0_floor = f0_floor
        self.f0_ceil = f0_ceil
        self.fft_length = fft_length
        self.dtype = dtype

    def _asdict(self):
        return self.__dict__
