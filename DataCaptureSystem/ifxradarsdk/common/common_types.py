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

"""Definitions of common structures for Infineon Radar sensors

This file contains definitions of enumerations and structures used by the wrapper.
"""

from ctypes import *
from enum import IntEnum

from .base_types import ifxStructure


class RadarSensor(IntEnum):
    """Radar sensors"""
    BGT60TR13C    = 0,   # BGT60TR13C
    BGT60ATR24C   = 1,   # BGT60ATR24C
    BGT60UTR13D   = 2,   # BGT60UTR13D
    BGT60TR12E    = 3,   # BGT60TR12E
    BGT60UTR11AIP    = 4,   # BGT60UTR11AIP
    BGT120UTR13E  = 5,   # BGT120UTR12E
    BGT24LTR24    = 6,   # BGT24LTR24
    BGT120UTR24   = 7,   # BGT120UTR24
    BGT60ATR24E   = 8,   # BGT60ATR24E
    BGT24LTR13E   = 9,   # BGT24LTR13E
    Unknown_Avian = 10,  # Unknown Avian device
    BGT24ATR22    = 128, # BGT24ATR22
    BGT60LTR11AIP = 256, # BGT60LTR11AIP
    Unknown_sensor= 4095 # Unknow sensor


class DeviceListEntry(ifxStructure):
    """Device list entry (ifx_Radar_Sensor_List_Entry_t)"""
    _fields_ = (("sensor_type", c_int),
                ("board_type", c_int),
                ("uuid", c_char * 37),
                )


class SensorInfo(ifxStructure):
    """Sensor info (ifx_Radar_Sensor_Info_t)"""
    _fields_ = (("description", c_char_p),
                ("min_rf_frequency_Hz", c_double),
                ("max_rf_frequency_Hz", c_double),
                ("num_tx_antennas", c_uint8),
                ("num_rx_antennas", c_uint8),
                ("max_tx_power", c_uint8),
                ("device_id", c_uint64),
                ("max_num_samples_per_chirp", c_uint32),
                ("min_adc_sampling_rate", c_float),
                ("max_adc_sampling_rate", c_float),
                ("adc_resolution_bits", c_uint8),
                ("lp_cutoff_list", POINTER(c_int32)),
                ("hp_cutoff_list", POINTER(c_int32)),
                ("if_gain_list", POINTER(c_int8)),
                )


class FirmwareInfo(ifxStructure):
    """Firmware information (ifx_Firmware_Info_t)"""
    _fields_ = (('description', c_char_p),
                ('version_major', c_uint16),
                ('version_minor', c_uint16),
                ('version_build', c_uint16),
                ('extended_version', c_char_p),
                )


# helper function to expand c arrays into lists
def create_python_list_from_terminated_list(pointer, terminator=-1) -> list:
    values = []
    if pointer:
        index = 0
        while (True):
            value = pointer[index]
            if (value == terminator):
                break  # end of list has been reached
            values.append(value)
            index += 1
    return values
