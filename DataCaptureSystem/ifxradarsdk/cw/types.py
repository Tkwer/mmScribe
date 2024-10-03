# ===========================================================================
# Copyright (C) 2021-2022 Infineon Technologies AG
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

"""Definitions of the various structures for the CW wrapper

This file contains definitions of enumerations and structures for the CW
wrapper.
"""

from ctypes import *
from enum import IntEnum

from ..common.common_types import ifxStructure


class BasebandConfig(ifxStructure):
    """Wrapper for structure ifx_Cw_Baseband_Config_t """
    _fields_ = (("start_frequency_Hz", c_double),
                ("end_frequency_Hz", c_double),
                ("sample_rate_Hz", c_float),
                ("num_samples", c_uint32),
                ("rx_mask", c_uint32),
                ("tx_mask", c_uint32),
                ("tx_power_level", c_uint32),
                ("lp_cutoff_Hz", c_int32),
                ("hp_cutoff_Hz", c_int32),
                ("if_gain_dB", c_int8),
                )


class AdcConfig(ifxStructure):
    """Wrapper for structure ifx_Cw_Adc_Config_t """
    _fields_ = (("sample_and_hold_time_ns", c_uint32),
                ("oversampling_factor", c_uint16),
                ("additional_subconversions", c_uint16),
                ("double_msb_time", c_bool),
                )


class DeviceTestSignalGeneratorMode(IntEnum):
    TEST_SIGNAL_MODE_OFF = 0
    TEST_SIGNAL_MODE_BASEBAND_TEST = 1
    TEST_SIGNAL_MODE_TOGGLE_TX_ENABLE = 2
    TEST_SIGNAL_MODE_TOGGLE_DAC_VALUE = 3
    TEST_SIGNAL_MODE_TOGGLE_RX_SELF_TEST = 4


class TestSignalGeneratorConfig(ifxStructure):
    """Wrapper for structure ifx_Cw_Test_Signal_Generator_Config_t """
    _fields_ = (("mode", c_uint32),
                ("frequency_Hz", c_float),
                )
