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

"""Definitions of the various structures for the LTR11 wrapper

This file contains definitions of enumerations and structures for the LTR11
wrapper.
"""
from ctypes import *

from ..common.base_types import ifxStructure


class Ltr11Config(ifxStructure):
    """Device configuration for LTR11 radar sensor

    mode: 
        Operating mode of the chip (Default is pulsed mode).

    rf_frequency_Hz:
        The RF center frequency in Hz.
    
    num_of_samples:
        The num_of_samples count representing the number of samples per frame per readout
        required. Allowed values are between 16 and 1024.
    
    detector_threshold:
        The detection threshold level. Allowed values are between 0 and 4095.

    prt:
        The pulse repetition time index, which corresponding value defines the time 
        until next pulsing sequence starts in pulsing.
        The get_prt_value method can be used to get the prt value in us from the prt index. 

    pulse_width:
        The pulse width corresponding to the time sampling is active during one pulsing event.  
    
    tx_power_level:
        The Medium power amplifier gain control.

    rx_if_gain:
        The Baseband PGA gain setting.
    
    aprt_factor:
        The Adaptive Pulse Repetition time factor. Initially the APRT is disabled.
        When enabled, the PRT is multiplied by this factor when no target is 
        detected.

    hold_time:
        The hold time for target detection. 

    disable_internal_detector:
        Flag indicating when true that the internal detector needs to be disabled.
    """
    _fields_ = (('mode', c_int),
                ('rf_frequency_Hz', c_uint64),
                ('num_of_samples', c_uint16),
                ('detector_threshold', c_uint16),
                ('prt', c_int),
                ('pulse_width', c_int),
                ('tx_power_level', c_int),
                ('rx_if_gain', c_int),
                ('aprt_factor', c_int),
                ('hold_time', c_int),
                ('disable_internal_detector', c_bool),
                )

    def get_prt_value(self):
        if self.prt == 0:
            return 250
        elif self.prt == 1:
            return 500
        elif self.prt == 2:
            return 1000
        elif self.prt == 3:
            return 2000
        else:
            raise ValueError


class RFCenterFrequencyHzLimits(ifxStructure):
    """RF Center Frequency limits structure."""
    _fields_ = (('min', c_uint64),
                ('max', c_uint64),
                ('step', c_uint64),
                )


class GenericLimits(ifxStructure):
    """ 
        Generic uint16_t limits structure for the values of  
        the detector_threshold and num_of_samples.
    """
    _fields_ = (('min', c_uint16),
                ('max', c_uint16),
                )


class Ltr11ConfigLimits(ifxStructure):
    """LTR11 Configuration Limits"""
    _fields_ = (('rf_frequency_Hz', RFCenterFrequencyHzLimits),
                ('num_of_samples', GenericLimits),
                ('detector_threshold', GenericLimits),
                )


class Ltr11Metadata(ifxStructure):
    """LTR11 digital detector output data"""
    _fields_ = (('avg_power', c_float),
                ('active', c_bool),
                ('motion', c_bool),
                ('direction', c_bool),
                )
