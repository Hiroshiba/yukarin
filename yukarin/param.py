class AcousticParam(object):
    def __init__(
            self,
            sampling_rate: int = 24000,
            pad_second: float = 0,
            frame_period: int = 5,
            order: int = 8,
            alpha: float = 0.466,
            f0_floor: float = 71,
            f0_ceil: float = 800,
            dtype: str = 'float32',
    ):
        self.sampling_rate = sampling_rate
        self.pad_second = pad_second
        self.frame_period = frame_period
        self.order = order
        self.alpha = alpha
        self.f0_floor = f0_floor
        self.f0_ceil = f0_ceil
        self.dtype = dtype
