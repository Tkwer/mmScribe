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

"""Python wrapper for Infineon Radar sensors CW (Continuous wave) operation"""

from ctypes import *
import typing

import numpy as np

from ..common.base_types import MdaReal
from ..common.cdll_helper import declare_prototype, load_library
from ..common.common_types import (
    create_python_list_from_terminated_list,
    DeviceListEntry,
    FirmwareInfo,
    RadarSensor,
    SensorInfo
)
from ..common.sdk_base import ifx_mda_destroy_r, move_ifx_list_to_python_list
from .types import AdcConfig, BasebandConfig, TestSignalGeneratorConfig


class DeviceCw():
    @staticmethod
    def __load_cdll() -> CDLL:
        """Initialize the module and return ctypes handle"""
        # find and load shared library
        dll = load_library("sdk_cw")

        # declare prototypes such that ctypes knows the arguments and return types
        declare_prototype(dll, "ifx_cw_create", None, c_void_p)
        declare_prototype(dll, "ifx_cw_get_list", None, c_void_p)
        declare_prototype(dll, "ifx_cw_create_by_uuid", [c_char_p], c_void_p)
        declare_prototype(dll, "ifx_cw_create_dummy", [c_int], c_void_p)
        declare_prototype(dll, "ifx_cw_create_dummy_from_device", [c_void_p], c_void_p)
        declare_prototype(dll, "ifx_cw_destroy", [c_void_p], None)
        declare_prototype(dll, "ifx_cw_save_register_file", [c_void_p, c_char_p], None)
        declare_prototype(dll, "ifx_cw_load_register_file", [c_void_p, c_char_p], None)
        declare_prototype(dll, "ifx_cw_is_signal_active", [c_void_p], c_bool)
        declare_prototype(dll, "ifx_cw_start_signal", [c_void_p], None)
        declare_prototype(dll, "ifx_cw_stop_signal", [c_void_p], None)
        declare_prototype(dll, "ifx_cw_set_baseband_config", [c_void_p, POINTER(BasebandConfig)], None)
        declare_prototype(dll, "ifx_cw_get_baseband_config", [c_void_p], POINTER(BasebandConfig))
        declare_prototype(dll, "ifx_cw_set_adc_config", [c_void_p, POINTER(AdcConfig)], None)
        declare_prototype(dll, "ifx_cw_get_adc_config", [c_void_p], POINTER(AdcConfig))
        declare_prototype(dll, "ifx_cw_set_test_signal_generator_config", [c_void_p, POINTER(TestSignalGeneratorConfig)], None)
        declare_prototype(dll, "ifx_cw_get_test_signal_generator_config", [c_void_p], POINTER(TestSignalGeneratorConfig))
        declare_prototype(dll, "ifx_cw_measure_temperature", [c_void_p], c_float)
        declare_prototype(dll, "ifx_cw_measure_tx_power", [c_void_p, c_uint32], c_float)
        declare_prototype(dll, "ifx_cw_capture_frame", [c_void_p, POINTER(MdaReal)], POINTER(MdaReal))
        declare_prototype(dll, "ifx_cw_get_sensor_information", [c_void_p], POINTER(SensorInfo))
        declare_prototype(dll, "ifx_cw_get_firmware_information", [c_void_p], POINTER(FirmwareInfo))

        return dll

    _cdll = __load_cdll.__func__()

    @classmethod
    def get_list(cls) -> typing.List[str]:
        """Return list of UUIDs of available boards

        The function returns a list of unique ids (uuids) that correspond to
        available boards.

        Note: boards which are already instantiated will not appear in the list.

        **Examples**
            uuids_all   = DeviceCw.get_list()
        """
        uuids = []

        ifx_list = cls._cdll.ifx_cw_get_list()

        return move_ifx_list_to_python_list(ifx_list,
                                            lambda p: cast(p, POINTER(DeviceListEntry)).contents.uuid.decode("ascii"))

    def __init__(self, uuid: typing.Optional[str] = None, sensor_type: typing.Optional[RadarSensor] = None,
                 handle: typing.Optional[c_void_p] = None):
        """Create and initialize CW controller

        Search for an Infineon radar sensor board connected to the host machine
        and connect to the first found sensor device.
        If a sensor_type is passed as a parameter, a dummy is created from this sensor type.

        The device is automatically closed by the destructor. If you want to
        close the device yourself, you can use the keyword del:
            device = DeviceCw()
            # do something with device
            ...
            # close device
            del device

        Optional parameters:
            uuid:       open the radar device with unique id given by uuid
                        the uuid is represented as a 32 character string of
                        hexadecimal characters. In addition, the uuid may
                        contain dash characters (-) which will be ignored.
                        Both examples are valid and correspond to the same
                        uuid:
                            0123456789abcdef0123456789abcdef
                            01234567-89ab-cdef-0123-456789abcdef
            sensor_type: Sensor of type RadarSensor
            handle:     creates a new instance from an existing handle (used internally)
        """

        if handle:
            self.handle = handle  # instantiate DeviceCw from an existing handle (e.g. dummy)
        else:
            if uuid:
                h = self._cdll.ifx_cw_create_by_uuid(uuid.encode("ascii"))
            elif sensor_type is not None:  # avoid ambiguity with RadarSensor.BGT60TR13C
                h = self._cdll.ifx_cw_create_dummy(sensor_type)
            else:
                h = self._cdll.ifx_cw_create()

            # The pointer is returned as an integer. Explicitly casting it to a void pointer
            # ensures it is not truncated by integer handling in certain situations
            self.handle = c_void_p(h)

    def create_dummy_from_device(self) -> 'DeviceCw':
        dummy_handle = self._cdll.ifx_cw_create_dummy_from_device(self.handle)
        return DeviceCw(handle=c_void_p(dummy_handle))

    def get_firmware_information(self) -> dict:
        """Gets information about the firmware of a connected device"""
        info_p = self._cdll.ifx_cw_get_firmware_information(self.handle)
        return info_p.contents.to_dict(True)

    def get_sensor_information(self) -> dict:
        """Gets information about the connected device"""
        info_p = self._cdll.ifx_cw_get_sensor_information(self.handle)
        info = info_p.contents.to_dict(True)

        for entry in ["lp_cutoff_list", "hp_cutoff_list", "if_gain_list"]:
            info[entry] = create_python_list_from_terminated_list(info[entry])

        return info

    def save_register_file(self, filename: str) -> None:
        """Save register list to a file"""
        filename_buffer = filename.encode("ascii")
        filename_buffer_p = c_char_p(filename_buffer)
        self._cdll.ifx_cw_save_register_file(self.handle, filename_buffer_p)

    def load_register_file(self, filename: str) -> None:
        """Load register list to a file"""
        filename_buffer = filename.encode("ascii")
        filename_buffer_p = c_char_p(filename_buffer)
        self._cdll.ifx_cw_load_register_file(self.handle, filename_buffer_p)

    def is_signal_active(self) -> bool:
        """Checks whether the continuous wave mode is enabled or not (returns boolean)"""
        signal_enabled = self._cdll.ifx_cw_is_signal_active(self.handle)
        return bool(signal_enabled)

    def start_signal(self) -> None:
        """Starts the continuous wave signal"""
        self._cdll.ifx_cw_start_signal(self.handle)

    def stop_signal(self) -> None:
        """Stops the continuous wave signal"""
        self._cdll.ifx_cw_stop_signal(self.handle)

    def set_baseband_config(self, baseband_config: BasebandConfig):
        """Sets the baseband configuration parameters

        Parameter:
        - baseband_config: the baseband parameters (BasebandConfig) to be set
        """
        self._cdll.ifx_cw_set_baseband_config(self.handle, byref(baseband_config))

    def get_baseband_config(self) -> BasebandConfig:
        """Gets the baseband configuration parameters"""
        conf_p = self._cdll.ifx_cw_get_baseband_config(self.handle)
        return conf_p.contents

    def set_adc_config(self, adc_config: AdcConfig):
        """Sets the ADC configuration parameters

        Parameter:
        - adc_config: the ADC parameters (ADCConfig) to be set
        """
        self._cdll.ifx_cw_set_adc_config(self.handle, byref(adc_config))

    def get_adc_config(self) -> AdcConfig:
        """Gets the ADC configuration parameters"""
        conf_p = self._cdll.ifx_cw_get_adc_config(self.handle)
        return conf_p.contents

    def set_test_signal_generator_config(self, signal_generator_config: TestSignalGeneratorConfig):
        """Sets the test signal generator configuration

        Parameter:
        - signal_generator_config: the signal generator configuration (TestSignalGeneratorConfig) to be set
        """
        self._cdll.ifx_cw_set_test_signal_generator_config(self.handle, byref(signal_generator_config))

    def get_test_signal_generator_config(self) -> TestSignalGeneratorConfig:
        """Gets the test signal generator configuration"""
        conf_p = self._cdll.ifx_cw_get_test_signal_generator_config(self.handle)
        return conf_p.contents

    def measure_temperature(self) -> float:
        """Get the temperature of the device in degrees Celsius"""
        temperature = self._cdll.ifx_cw_measure_temperature(self.handle)
        return float(temperature)

    def measure_tx_power(self, antenna) -> float:
        """Gets the tx power level
        Return power is equal to -1 if CW signal is not active

        Parameter:
        - antenna: index of the antenna to be measured. The value
                   is 0 based and must be less than the value \ref num_tx_antennas
                   returned by \ref ifx_cw_get_sensor_information. If the value is
                   not in the allowed range, an exception is thrown."""
        tx_power = self._cdll.ifx_cw_measure_tx_power(self.handle, antenna)
        return float(tx_power)

    def capture_frame(self) -> np.ndarray:
        """Captures a single frame in CW mode and returns it as a 2D numpy array"""
        frame = self._cdll.ifx_cw_capture_frame(self.handle, None)
        frame_numpy = frame.contents.to_numpy()
        ifx_mda_destroy_r(frame)
        return frame_numpy

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._close()

    def _close(self):
        """Destroy device handle"""
        if hasattr(self, "handle") and self.handle:
            self._cdll.ifx_cw_destroy(self.handle)
            self.handle = None

    def __del__(self):
        try:
            self._close()
        except:
            # just like in C++ exceptions raised in a destructor cannot be catched
            # for this reason we ignore them
            pass
