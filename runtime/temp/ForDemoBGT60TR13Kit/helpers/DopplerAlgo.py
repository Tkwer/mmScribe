# ===========================================================================
# Copyright (C) 2022 Infineon Technologies AG
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
# ===========================================================================

import numpy as np
from scipy import signal

from helpers.fft_spectrum import *


class DopplerAlgo:
    """Compute Range-Doppler map"""

    def __init__(self, num_samples: int, num_chirps_per_frame: int, num_ant: int, mti_alpha: float = 0.6):
        """Create Range-Doppler map object

        Parameters:
            - num_samples:          Number of samples in a single chirp
            - num_chirps_per_frame: Number of chirp repetitions within a measurement frame
            - num_ant:              Number of antennas
            - mti_alpha:            Parameter alpha of Moving Target Indicator
        """
        self.num_chirps_per_frame = num_chirps_per_frame

        # compute Blackman-Harris Window matrix over chirp samples(range)
        self.range_window = signal.hanning(num_samples).reshape(1, num_samples)

        # compute Blackman-Harris Window matrix over number of chirps(velocity)
        self.doppler_window = signal.hanning(self.num_chirps_per_frame).reshape(1, self.num_chirps_per_frame)

        # parameter for moving target indicator (MTI)
        self.mti_alpha = mti_alpha

        # initialize MTI filter
        self.mti_history = np.zeros((self.num_chirps_per_frame, num_samples, num_ant), dtype=np.float32)

    def compute_doppler_map(self, data: np.ndarray, i_ant: int):
        """Compute Range-Doppler map for i-th antennas

        Parameter:
            - data:     Raw-data for one antenna (dimension:
                        num_chirps_per_frame x num_samples)
            - i_ant:    RX antenna index
        """
        # Step 1 - Remove average from signal (mean removal)
        # data = data - np.average(data)

        # Step 2 - MTI processing to remove static objects
        data_mti = data - self.mti_history[:, :, i_ant]
        self.mti_history[:, :, i_ant] = data * self.mti_alpha + self.mti_history[:, :, i_ant] * (1 - self.mti_alpha)

        # Step 3 - calculate fft spectrum for the frame
        fft1d = fft_spectrum(data_mti, self.range_window)

        # prepare for doppler FFT

        # Transpose
        # Distance is now indicated on y axis
        fft1d = np.transpose(fft1d)

        # Step 4 - Windowing the Data in doppler
        fft1d = np.multiply(fft1d, self.doppler_window)

        # zp2 = np.pad(fft1d, ((0, 0), (0, self.num_chirps_per_frame)), "constant")
        # 不进行补0 操作了
        fft2d = np.fft.fft(fft1d) / self.num_chirps_per_frame

        # re-arrange fft result for zero speed at centre
        return np.fft.fftshift(fft2d, (1,))
