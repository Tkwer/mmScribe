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

"""Definitions of the various structures for the Mimose wrapper

This file contains definitions of enumerations and structures for the Mimose
wrapper.
"""

from ctypes import *
from enum import IntEnum

from ..common.base_types import ifxStructure


class ifx_Mimose_Channel_t(IntEnum):
    """Lists all building blocks a frame sequence can be built from"""
    IFX_MIMOSE_CHANNEL_TX1_RX1 = 0
    IFX_MIMOSE_CHANNEL_TX2_RX2 = 1
    IFX_MIMOSE_CHANNEL_TX1_RX2 = 2
    IFX_MIMOSE_CHANNEL_TX2_RX1 = 3


class ifx_Mimose_ABB_type_t(IntEnum):
    """Lists all building blocks a frame sequence can be built from"""
    IFX_MIMOSE_ABB_GAIN_3          = 0
    IFX_MIMOSE_ABB_GAIN_6          = 1
    IFX_MIMOSE_ABB_GAIN_12         = 2
    IFX_MIMOSE_ABB_GAIN_24         = 3
    IFX_MIMOSE_ABB_GAIN_48         = 4
    IFX_MIMOSE_ABB_GAIN_96         = 5
    IFX_MIMOSE_ABB_GAIN_192        = 6
    IFX_MIMOSE_ABB_GAIN_384        = 7
    IFX_MIMOSE_ABB_GAIN_AUTOMATIC  = 255


class ifx_Mimose_AOC_Mode_t(IntEnum):
    """Lists all building blocks a frame sequence can be built from"""
    IFX_MIMOSE_AOC_OFFSET_VALUE_IS_APPLIED                     = 0
    IFX_MIMOSE_AOC_OFFSET_VALUE_IS_ONLY_TRACKED                = 1
    IFX_MIMOSE_AOC_OFFSET_VALUE_IS_FULLY_HANDLED_BY_THE_FSM_2  = 2
    IFX_MIMOSE_AOC_OFFSET_VALUE_IS_FULLY_HANDLED_BY_THE_FSM_3  = 3


class ifx_Mimose_Pulse_Config_t(ifxStructure):
    """Wrapper for structure ifx_Mimose_Pulse_Config_t."""
    _fields_ = (("channel", c_uint32),
                ("tx_power_level", c_uint8),
                ("abb_gain_type", c_int),
                ("aoc_mode", c_int))


class ifx_Mimose_Frame_Config_t(ifxStructure):
    """Wrapper for structure ifx_Mimose_Frame_Config_t."""
    _fields_ = (("frame_repetition_time_s", c_float),
                ("pulse_repetition_time_s", c_float),
                ("selected_pulse_configs", (c_bool * 4)),
                ("num_of_samples", c_uint16))


class ifx_Mimose_AFC_Control_t(ifxStructure):
    """Wrapper for structure ifx_Mimose_AFC_Control_t."""
    _fields_ = (("band", c_int),
                ("rf_center_frequency_Hz", c_uint64),
                ("afc_duration_ct", c_uint16),
                ("afc_threshold_course", c_uint16),
                ("afc_threshold_fine", c_uint16),
                ("afc_period", c_uint8),
                ("afc_repeat_count", c_uint32))


class ifx_Mimose_Clock_Config_t(ifxStructure):
    """Mimose Clock Configuration"""
    _fields_ = (('reference_clock_Hz', c_uint32),
                ('system_clock_Hz', c_uint32),
                ('rc_clock_enabled', c_bool),
                ('hf_on_time_usec', c_uint32),
                ('system_clock_divider', c_uint16),
                ('system_clock_div_flex', c_bool),
                ('sys_clk_to_i2c', c_bool),
                )


class ifx_Mimose_Config_t(Structure):
    """Wrapper for structure ifx_Mimose_Config_t."""
    _fields_ = (("PulseConfig", (ifx_Mimose_Pulse_Config_t * 4)),
                ("FrameConfig", (ifx_Mimose_Frame_Config_t * 2)),
                ("AFC_Config", ifx_Mimose_AFC_Control_t),
                ("ClockConfig", ifx_Mimose_Clock_Config_t))


class MimoseMetadata(ifxStructure):
    """Wrapper for structure ifx_Mimose_Metadata_t."""
    _fields_ = (("abb_gains", (c_uint8 * 4)),
                ("aoc_offsets", ((c_short * 2) * 4)))


class ifx_Mimose_ABB_type_t(IntEnum):
    IFX_MIMOSE_ABB_GAIN_3 = 0
    IFX_MIMOSE_ABB_GAIN_6 = 1
    IFX_MIMOSE_ABB_GAIN_12 = 2
    IFX_MIMOSE_ABB_GAIN_24 = 3
    IFX_MIMOSE_ABB_GAIN_48 = 4
    IFX_MIMOSE_ABB_GAIN_96 = 5
    IFX_MIMOSE_ABB_GAIN_192 = 6
    IFX_MIMOSE_ABB_GAIN_384 = 7
    IFX_MIMOSE_ABB_GAIN_AUTOMATIC = 255


class ifx_Mimose_AFC_Repeat_Count_t(IntEnum):
    IFX_MIMOSE_AFC_REPEAT_COUNT_1 = 0
    IFX_MIMOSE_AFC_REPEAT_COUNT_2 = 1
    IFX_MIMOSE_AFC_REPEAT_COUNT_4 = 2
    IFX_MIMOSE_AFC_REPEAT_COUNT_8 = 3


class ifx_Mimose_AOC_Mode_t(IntEnum):
    IFX_MIMOSE_AOC_OFFSET_VALUE_IS_APPLIED = 0
    IFX_MIMOSE_AOC_OFFSET_VALUE_IS_ONLY_TRACKED = 1
    IFX_MIMOSE_AOC_OFFSET_VALUE_IS_FULLY_HANDLED_BY_THE_FSM_2 = 2
    IFX_MIMOSE_AOC_OFFSET_VALUE_IS_FULLY_HANDLED_BY_THE_FSM_3 = 3


class ifx_Mimose_RF_Band_t(IntEnum):
    IFX_MIMOSE_RF_BAND_ISM_SRR_24GHZ_100MHz = 0
    IFX_MIMOSE_RF_BAND_ISM_SRR_24GHZ_200MHz = 1
