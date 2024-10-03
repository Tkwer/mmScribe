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

"""Python wrapper for Infineon LTR11 Radar sensor"""

from ctypes import *
import typing

import numpy as np

from ..common.base_types import MdaComplex
from ..common.cdll_helper import declare_prototype, load_library
from ..common.common_types import (
    create_python_list_from_terminated_list,
    DeviceListEntry,
    FirmwareInfo,
    SensorInfo
)
from ..common.sdk_base import ifx_mda_destroy_c, move_ifx_list_to_python_list
from .types import (
    GenericLimits,
    Ltr11Config,
    Ltr11ConfigLimits,
    Ltr11Metadata,
    RFCenterFrequencyHzLimits
)


class DeviceLtr11():
    @staticmethod
    def __load_cdll() -> CDLL:
        """Initialize the module and return ctypes handle"""
        # find and load shared library
        dll = load_library("sdk_ltr11")

        # declare prototypes such that ctypes knows the arguments and return types
        declare_prototype(dll, "ifx_ltr11_create", None, c_void_p)
        declare_prototype(dll, "ifx_ltr11_destroy", [c_void_p], None)
        declare_prototype(dll, "ifx_ltr11_create_by_uuid", [c_void_p], c_void_p)
        declare_prototype(dll, "ifx_ltr11_get_list", None, c_void_p)
        declare_prototype(dll, "ifx_ltr11_get_config_defaults", [c_void_p, POINTER(Ltr11Config)], None)
        declare_prototype(dll, "ifx_ltr11_get_config", [c_void_p, POINTER(Ltr11Config)], None)
        declare_prototype(dll, "ifx_ltr11_set_config", [c_void_p, POINTER(Ltr11Config)], None)
        declare_prototype(dll, "ifx_ltr11_start_acquisition", [c_void_p], None)
        declare_prototype(dll, "ifx_ltr11_stop_acquisition", [c_void_p], None)
        declare_prototype(dll, "ifx_ltr11_get_next_frame", [c_void_p, POINTER(MdaComplex), POINTER(Ltr11Metadata)], POINTER(MdaComplex))
        declare_prototype(dll, "ifx_ltr11_get_next_frame_timeout", [c_void_p, POINTER(MdaComplex), POINTER(Ltr11Metadata), c_uint16], POINTER(MdaComplex))
        declare_prototype(dll, "ifx_ltr11_get_sensor_information", [c_void_p], POINTER(SensorInfo))
        declare_prototype(dll, "ifx_ltr11_get_firmware_information", [c_void_p], POINTER(FirmwareInfo))
        declare_prototype(dll, "ifx_ltr11_get_active_mode_power", [c_void_p, POINTER(Ltr11Config)], c_float)
        declare_prototype(dll, "ifx_ltr11_get_sampling_frequency", [c_void_p, c_int], c_uint32)
        declare_prototype(dll, "ifx_ltr11_check_config", [c_void_p, POINTER(Ltr11Config)], c_bool)
        declare_prototype(dll, "ifx_ltr11_register_dump_to_file", [c_void_p, c_char_p], None)
        declare_prototype(dll, "ifx_ltr11_get_limits", [c_void_p, POINTER(Ltr11ConfigLimits)], None)

        return dll

    _cdll = __load_cdll.__func__()

    @classmethod
    def get_list(cls) -> typing.List[str]:
        """Return list of UUIDs of available boards

        The function returns a list of unique ids (uuids) that correspond to
        available boards.

        Note: boards which are already instantiated will not appear in the list.

        **Examples**
            uuids_all   = Device.get_list()
        """

        ifx_list = cls._cdll.ifx_ltr11_get_list()

        return move_ifx_list_to_python_list(ifx_list,
                                            lambda p: cast(p, POINTER(DeviceListEntry)).contents.uuid.decode("ascii"))

    def __init__(self, uuid: typing.Optional[str] = None):
        """Create and initialize Ltr11 controller

        Search for an Infineon radar sensor board connected to the host machine
        and connect to the first found sensor device.

        The device is automatically closed by the destructor. If you want to
        close the device yourself, you can use the keyword del:
            device = DeviceLtr11()
            # do something with device
            ...
            # close device
            del device

        However, we suggest to use a context manager:
            with open DeviceLtr11() as device:
                # do something with device

        If uuid is given the radar device with the given uuid is opened. If
        no parameters are given, the first found radar device will be opened.

        Examples:
          - Open first found radar device:
            device = DeviceLtr11()
          - Open ltr11 doppler radar device with uuid 0123456789abcdef0123456789abcdef
            device = DeviceLtr11(uuid="0123456789abcdef0123456789abcdef")

        Optional parameters:
            uuid:       open the ltr11 doppler radar device with unique id given by uuid.
                        The uuid is represented as a 32 character string of hexadecimal
                        characters. In addition, the uuid may contain dash characters (-),
                        which will be ignored.
                        Both examples are valid and correspond to the same
                        uuid:
                            0123456789abcdef0123456789abcdef
                            01234567-89ab-cdef-0123-456789abcdef
        """

        if uuid:
            h = self._cdll.ifx_ltr11_create_by_uuid(uuid.encode("ascii"))
        else:
            h = self._cdll.ifx_ltr11_create()

        # The pointer is returned as an integer. Explicitly casting it to a void pointer
        # ensures it is not truncated by integer handling in certain situations
        self.handle = c_void_p(h)

    def get_firmware_information(self) -> dict:
        """Gets information about the firmware of a connected device"""
        info_p = self._cdll.ifx_ltr11_get_firmware_information(self.handle)
        return info_p.contents.to_dict(True)

    def get_sensor_information(self) -> dict:
        """Gets information about the connected device"""
        info_p = self._cdll.ifx_ltr11_get_sensor_information(self.handle)
        info = info_p.contents.to_dict(True)

        for entry in ["lp_cutoff_list", "hp_cutoff_list", "if_gain_list"]:
            info[entry] = create_python_list_from_terminated_list(info[entry])

        return info

    def set_config(self, config: Ltr11Config) -> None:
        """Set LTR11 configuration"""
        self._cdll.ifx_ltr11_set_config(self.handle, byref(config))

    def get_config(self) -> Ltr11Config:
        """Get current LTR11 configuration"""
        config = Ltr11Config(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False)
        self._cdll.ifx_ltr11_get_config(self.handle, byref(config))
        return config

    def get_config_defaults(self) -> Ltr11Config:
        """Get default device configuration"""
        config = Ltr11Config(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False)
        self._cdll.ifx_ltr11_get_config_defaults(self.handle, byref(config))
        return config

    def get_limits(self) -> Ltr11ConfigLimits:
        """Get LTR11 configuration limits"""
        limits = Ltr11ConfigLimits(RFCenterFrequencyHzLimits(
            0, 0), GenericLimits(0, 0), GenericLimits(0, 0))
        self._cdll.ifx_ltr11_get_limits(self.handle, byref(limits))
        return limits

    def check_config(self, config: Ltr11Config) -> bool:
        """ Check if the config is valid, and return true if the config 
         is valid or false otherwise.
        """
        config_valid = self._cdll.ifx_ltr11_check_config(self.handle, byref(config))
        return bool(config_valid)

    def start_acquisition(self) -> None:
        """Start acquisition of time domain data

        Start the acquisition of time domain data from the connected device.
        If the data acquisition is already running the function has no effect.
        """
        self._cdll.ifx_ltr11_start_acquisition(self.handle)

    def stop_acquisition(self) -> None:
        """Stop acquisition of time domain data

        Stop the acquisition of time domain data from the connected device.
        If the data acquisition is already stopped the function has no effect.
        """
        self._cdll.ifx_ltr11_stop_acquisition(self.handle)

    def get_next_frame(self, timeout_ms: typing.Optional[int] = None) -> typing.Tuple[np.ndarray, Ltr11Metadata]:
        """Retrieve next frame of time domain data from LTR11 device.

        Retrieve the next complete frame of time domain data from the connected
        device. The samples from all chirps and all enabled RX antennas will be
        copied to the provided data structure frame.

        The frame is returned as numpy array with dimensions the num_of_samples, 
        which is a member variable in the Ltr11Config structure. 
        The metadata of type Ltr11Metadata is returned. 

        If timeout_ms is given, the exception ErrorTimeout is raised if a
        complete frame is not available within timeout_ms milliseconds.
        """
        metadata = Ltr11Metadata()

        if timeout_ms:
            frame = self._cdll.ifx_ltr11_get_next_frame_timeout(
                self.handle, None, byref(metadata), timeout_ms)
        else:
            frame = self._cdll.ifx_ltr11_get_next_frame(
                self.handle, None, byref(metadata))

        frame_numpy = frame.contents.to_numpy()
        ifx_mda_destroy_c(frame)

        return frame_numpy, metadata

    def get_active_mode_power(self, config: Ltr11Config) -> float:
        """ Return the power in active mode for a given configuration. 
            i.e when the APRT (Adaptive prt LTR11 feature) is disabled,
            or when the latter feature is enabled, and a target is detected. 
            In case, the APRT is enabled and no target a detected, the power
            consumption is lower, hence, the chip is in low-power mode.
            The average power (avg_power, member of the Ltr11Metadata) 
        """
        power_level = self._cdll.ifx_ltr11_get_active_mode_power(self.handle, config)
        return float(power_level)

    def get_sampling_frequency(self, prt_index: int) -> int:
        """ Return the the sampling frequency in Hz for the given prt_index 

            In case the APRT feature of the chip is enabled and no target is detected, 
            the pulse repetition time is multiplied by the aprt_factor.
             Hence, the sampling frequency is divided by this factor. 
            This function does not include this specific case, and only returns the sampling
            frequency value computed from the prt when the chip is in active mode.
        """
        sampling_frequency_Hz = self._cdll.ifx_ltr11_get_sampling_frequency(self.handle, prt_index)
        return int(sampling_frequency_Hz)

    def register_dump_to_file(self, filename: str) -> None:
        """Dump register list to a file"""
        filename_buffer = filename.encode("ascii")
        filename_buffer_p = c_char_p(filename_buffer)
        self._cdll.ifx_ltr11_register_dump_to_file(self.handle, filename_buffer_p)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._close()

    def _close(self):
        """Destroy device handle"""
        if hasattr(self, "handle") and self.handle:
            self._cdll.ifx_ltr11_destroy(self.handle)
            self.handle = None

    def __del__(self):
        try:
            self._close()
        except:
            # just like in C++ exceptions raised in a destructor cannot be catched
            # for this reason we ignore them
            pass
