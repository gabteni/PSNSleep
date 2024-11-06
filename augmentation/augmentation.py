from audiomentations.core.transforms_interface import BaseWaveformTransform
from numpy.typing import NDArray
import random

import numpy as np
class RandomMask(BaseWaveformTransform):
    

    supports_multichannel = True

    def __init__(
        self,
        min_band_part: float = 0.0,
        max_band_part: float = 0.5,
        fade: bool = False,
        p: float = 0.05,
        pr: float = 0.05,
    ):
        """
        :param min_band_part: Minimum length of the silent part as a fraction of the
            total sound length. Must be between 0.0 and 1.0
        :param max_band_part: Maximum length of the silent part as a fraction of the
            total sound length. Must be between 0.0 and 1.0
        :param fade: When set to True, add a linear fade in and fade out of the silent
            part. This can smooth out an unwanted abrupt change between two consecutive
            samples (which sounds like a transient/click/pop).
        :param p: The probability of applying this transform
        """
        super().__init__(p)
        if min_band_part < 0.0 or min_band_part > 1.0:
            raise ValueError("min_band_part must be between 0.0 and 1.0")
        if max_band_part < 0.0 or max_band_part > 1.0:
            raise ValueError("max_band_part must be between 0.0 and 1.0")
        if min_band_part > max_band_part:
            raise ValueError("min_band_part must not be greater than max_band_part")
        self.min_band_part = min_band_part
        self.max_band_part = max_band_part
        self.fade = fade
        self.pr=pr
    def randomize_parameters(self, samples: NDArray[np.float32], sample_rate: int):
        super().randomize_parameters(samples, sample_rate)
        if self.parameters["should_apply"]:
            num_samples = samples.shape[-1]
            self.parameters["t"] = random.randint(
                int(num_samples * self.min_band_part),
                int(num_samples * self.max_band_part),
            )
            self.parameters["t0"] = random.randint(
                0, num_samples - self.parameters["t"]
            )

    def apply(self, samples: NDArray[np.float32], sample_rate: int):
        new_samples = samples.copy()
        
        t = self.parameters["t"]
        t0 = self.parameters["t0"]
        mask = np.zeros(t)
        ##########
        
        mask = (np.random.rand(*samples.shape) < self.pr).astype(int)
        
        mask=np.cumsum(mask,axis=-1) % 2
        #print(mask.shape)      

        new_samples=new_samples*mask
        """if self.fade:
            fade_length = min(int(sample_rate * 0.01), int(t * 0.1))
            mask[0:fade_length] = np.linspace(1, 0, num=fade_length)
            mask[-fade_length:] = np.linspace(0, 1, num=fade_length)
        new_samples[..., t0 : t0 + t] *= mask"""
        #print(new_samples.shape) 
        return new_samples