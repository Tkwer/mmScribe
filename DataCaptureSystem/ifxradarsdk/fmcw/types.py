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

"""Definitions of the various structures for the FMCW wrapper

This file contains definitions of enumerations and structures for the FMCW
wrapper.
"""

from ctypes import *
from enum import IntEnum

from ..common.base_types import MdaReal, ifxStructure


class FmcwFrame(ifxStructure):
    """Wrapper for structure ifx_Fmcw_Frame_t"""
    _fields_ = (("num_cubes", c_uint32),
                ("cubes", POINTER(POINTER(MdaReal))),
                )


class FmcwElementType(IntEnum):
    """Lists all building blocks a frame sequence can be built from"""
    IFX_SEQ_LOOP = 0
    IFX_SEQ_CHIRP = 1
    IFX_SEQ_DELAY = 2


class FmcwSequenceElement(ifxStructure):
    """Wrapper for structure ifx_Fmcw_Sequence_Element_t"""
    pass  # fields must be initialised below due to recursiveness


class FmcwSequenceLoop(ifxStructure):
    """Wrapper for structure ifx_Fmcw_Sequence_Loop_t"""
    _fields_ = (("sub_sequence", POINTER(FmcwSequenceElement)),
                ("num_repetitions", c_uint32),
                ("repetition_time_s", c_float),
                )


class FmcwSequenceChirp(ifxStructure):
    """Wrapper for structure ifx_Fmcw_Sequence_Chirp_t"""
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


class FmcwSequenceDelay(ifxStructure):
    """Wrapper for structure ifx_Fmcw_Sequence_Delay_t"""
    _fields_ = (("time_s", c_float),
                )


class FmcwSequenceElementUnion(Union):
    _fields_ = (("loop", FmcwSequenceLoop),
                ("chirp", FmcwSequenceChirp),
                ("delay", FmcwSequenceDelay),
                )


# structure must be declared above due to recursiveness
#class FmcwSequenceElement(ifxStructure):
#    """Wrapper for structure ifx_Fmcw_Sequence_Element_t"""
FmcwSequenceElement._anonymous_ = ("union",)
FmcwSequenceElement._fields_ = [
    ("type", c_uint32),
    ("next_element", POINTER(FmcwSequenceElement)),
    ("union", FmcwSequenceElementUnion),
    ]


class FmcwSimpleSequenceConfig(ifxStructure):
    """Wrapper for structure ifx_Fmcw_Simple_Sequence_Config_t"""
    _fields_ = (("frame_repetition_time_s", c_float),
                ("chirp_repetition_time_s", c_float),
                ("num_chirps", c_uint32),
                ("tdm_mimo", c_bool),
                ("chirp", FmcwSequenceChirp),
                )


class FmcwMetrics(ifxStructure):
    """Wrapper for structure ifx_Fmcw_Metrics_t"""
    _fields_ = (("range_resolution_m", c_float),
                ("max_range_m", c_float),
                ("max_speed_m_s", c_float),
                ("speed_resolution_m_s", c_float),
                ("center_frequency_Hz", c_double)
                )


def create_dict_from_sequence_recursive(first_element: FmcwSequenceElement) -> dict:
    sequence = list()
    element = first_element
    while (True):
        type = FmcwElementType(element.type)
        if type is FmcwElementType.IFX_SEQ_LOOP:
            loop_dict = element.loop.to_dict(True)
            if element.loop.sub_sequence is False:
                raise ValueError("FmcwSequenceLoop sub_sequence maps to a null pointer")
            loop_dict.update(sub_sequence=create_dict_from_sequence_recursive(element.loop.sub_sequence.contents))
            sequence.append(dict(loop=loop_dict))
        elif type is FmcwElementType.IFX_SEQ_CHIRP:
            sequence.append(dict(chirp=element.chirp.to_dict(True)))
        elif type is FmcwElementType.IFX_SEQ_DELAY:
            sequence.append(dict(delay=element.delay.to_dict(True)))
        else:
            raise ValueError("Invalid FmcwSequenceElement type")

        if element.next_element:
            element = element.next_element.contents
        else:
            break  # end of sequence has been reached

    return sequence


# helper function to expand sequence c structure into dictionary
def create_dict_from_sequence(first_element: FmcwSequenceElement) -> dict:
    return create_dict_from_sequence_recursive(first_element)
