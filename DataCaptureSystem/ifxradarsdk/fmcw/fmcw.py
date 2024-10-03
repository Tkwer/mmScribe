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

"""Python wrapper for Infineon Radar sensors FMCW (frequency modulated continuous wave) operation"""

from ctypes import *
import typing

import numpy as np

from ..common.cdll_helper import declare_prototype, load_library
from ..common.common_types import (
    create_python_list_from_terminated_list,
    DeviceListEntry,
    FirmwareInfo,
    RadarSensor,
    SensorInfo
)
from ..common.sdk_base import move_ifx_list_to_python_list
from .types import (
    FmcwElementType,
    FmcwFrame,
    FmcwMetrics,
    FmcwSequenceChirp,
    FmcwSequenceElement,
    FmcwSimpleSequenceConfig
)


class DeviceFmcw():
    @staticmethod
    def __load_cdll() -> CDLL:
        """Initialize the module and return ctypes handle"""
        # find and load shared library
        dll = load_library("sdk_fmcw")

        # declare prototypes such that ctypes knows the arguments and return types
        declare_prototype(dll, "ifx_fmcw_get_list", None, c_void_p)
        declare_prototype(dll, "ifx_fmcw_get_list_by_sensor_type", [c_int], c_void_p)
        declare_prototype(dll, "ifx_fmcw_create_simple_sequence", [POINTER(FmcwSimpleSequenceConfig)], POINTER(FmcwSequenceElement))
        declare_prototype(dll, "ifx_fmcw_metrics_from_sequence", [POINTER(FmcwSequenceElement), POINTER(FmcwMetrics)], None)
        declare_prototype(dll, "ifx_fmcw_sequence_from_metrics", [POINTER(FmcwMetrics), c_bool, POINTER(FmcwSequenceElement)], None)
        declare_prototype(dll, "ifx_fmcw_create", None, c_void_p)
        declare_prototype(dll, "ifx_fmcw_create_by_uuid", [c_char_p], c_void_p)
        declare_prototype(dll, "ifx_fmcw_create_by_port", [c_char_p], c_void_p)
        declare_prototype(dll, "ifx_fmcw_create_dummy", [c_void_p], c_void_p)
        declare_prototype(dll, "ifx_fmcw_create_dummy_from_device", [c_void_p], c_void_p)
        declare_prototype(dll, "ifx_fmcw_destroy", [c_void_p], None)
        declare_prototype(dll, "ifx_fmcw_save_register_file", [c_void_p, c_char_p], None)
        declare_prototype(dll, "ifx_fmcw_load_register_file", [c_void_p, c_char_p], None)
        declare_prototype(dll, "ifx_fmcw_set_acquisition_sequence", [c_void_p, POINTER(FmcwSequenceElement)], None)
        declare_prototype(dll, "ifx_fmcw_get_acquisition_sequence", [c_void_p], POINTER(FmcwSequenceElement))
        declare_prototype(dll, "ifx_fmcw_get_board_uuid", [c_void_p], c_char_p)
        declare_prototype(dll, "ifx_fmcw_get_sensor_type", [c_void_p], RadarSensor)
        declare_prototype(dll, "ifx_fmcw_get_sensor_information", [c_void_p], POINTER(SensorInfo))
        declare_prototype(dll, "ifx_fmcw_get_firmware_information", [c_void_p], POINTER(FirmwareInfo))
        declare_prototype(dll, "ifx_fmcw_get_temperature", [c_void_p], c_float)
        declare_prototype(dll, "ifx_fmcw_start_acquisition", [c_void_p], None)
        declare_prototype(dll, "ifx_fmcw_stop_acquisition", [c_void_p], None)
        declare_prototype(dll, "ifx_fmcw_get_next_frame", [c_void_p, POINTER(FmcwFrame)], None)
        declare_prototype(dll, "ifx_fmcw_get_next_frame_timeout", [c_void_p, POINTER(FmcwFrame), c_uint16], None)
        declare_prototype(dll, "ifx_fmcw_allocate_frame", [c_void_p], POINTER(FmcwFrame))
        declare_prototype(dll, "ifx_fmcw_destroy_frame", [POINTER(FmcwFrame)], None)
        declare_prototype(dll, "ifx_fmcw_get_element_duration", [c_void_p, POINTER(FmcwSequenceElement)], c_float)
        declare_prototype(dll, "ifx_fmcw_get_sequence_duration", [c_void_p, POINTER(FmcwSequenceElement)], c_float)
        declare_prototype(dll, "ifx_fmcw_get_minimum_chirp_repetition_time", [c_void_p, c_uint32, c_float], c_float)
        declare_prototype(dll, "ifx_fmcw_get_chirp_sampling_bandwidth", [c_void_p, POINTER(FmcwSequenceChirp)], c_double)
        declare_prototype(dll, "ifx_fmcw_get_chirp_sampling_center_frequency", [c_void_p, POINTER(FmcwSequenceChirp)], c_double)

        return dll

    _cdll = __load_cdll.__func__()

    @classmethod
    def get_list(cls, sensor_type: typing.Optional[RadarSensor] = None) -> typing.List[str]:
        """Return list of UUIDs of available boards

        The function returns a list of unique ids (uuids) that correspond to
        available boards. If sensor_type is specified, only boards with this particular
        sensor are listed.

        Note: boards which are already instantiated will not appear in the list.

        **Examples**
            uuids_all   = DeviceFmcw.get_list()
            uuids_tr13c = DeviceFmcw.get_list(RadarSensor.BGT60TR13C)

        Parameters:
            sensor_type: Sensor of type RadarSensor
        """
        uuids = []

        if sensor_type == None:
            ifx_list = cls._cdll.ifx_fmcw_get_list()
        else:
            ifx_list = cls._cdll.ifx_fmcw_get_list_by_sensor_type(int(sensor_type))

        return move_ifx_list_to_python_list(ifx_list,
                                            lambda p: cast(p, POINTER(DeviceListEntry)).contents.uuid.decode("ascii"))

    @classmethod
    def create_simple_sequence(cls, config: FmcwSimpleSequenceConfig) -> FmcwSequenceElement:
        """This function initializes a single shape configuration structure, setting
         each field to a defined value and linking the loops and chirp members
         properly."""
        first_element = cls._cdll.ifx_fmcw_create_simple_sequence(byref(config))
        return first_element.contents

    @classmethod
    def metrics_from_sequence(cls, chirp_loop: FmcwSequenceElement) -> FmcwMetrics:
        """This function extracts metrics parameters from the provided chirp loop sequence."""
        metrics = FmcwMetrics()
        cls._cdll.ifx_fmcw_metrics_from_sequence(byref(chirp_loop), byref(metrics))
        return metrics

    @classmethod
    def sequence_from_metrics(cls, metrics: FmcwMetrics, chirp_loop: FmcwSequenceElement, round_to_power_of_2: bool = True) -> None:
        """Determines chirp loop sequence out of metric parameters

         If round_to_power_of_2 is True, the computed values for
         num_samples_per_chirp and num_chirps_per_frame will be rounded to the next
         power of 2.

         This function is not filling the following parameters of device config structure and caller of this function shall 
         fill these values from existing device config.
         - sample_rate_Hz
         - tx_mask
         - rx_mask
         - tx_power_level
         - lp_cutoff_Hz
         - hp_cutoff_Hz
         - if_gain_dB
        """
        cls._cdll.ifx_fmcw_sequence_from_metrics(byref(metrics), c_bool(round_to_power_of_2), byref(chirp_loop))

    def __init__(self, uuid: typing.Optional[str] = None, port: typing.Optional[str] = None,
                 sensor_type: typing.Optional[RadarSensor] = None, handle: typing.Optional[c_void_p] = None):
        """Create and initialize FMCW controller

        Search for an Infineon radar sensor board connected to the host machine
        and connect to the first found sensor device.
        If a sensor_type is passed as a parameter, a dummy is created from this sensor type.

        The device is automatically closed by the destructor. If you want to
        close the device yourself, you can use the keyword del:
            device = DeviceFmcw()
            # do something with device
            ...
            # close device
            del device

        Optional parameters:
            port:       opens the given port
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

         Examples:
          - Open first found radar device:
            dev = DeviceFmcw()
          - Open radar device on COM5:
            dev = DeviceFmcw(port="COM5")
          - Open radar device with uuid 0123456789abcdef0123456789abcdef
            dev = DeviceFmcw(uuid="0123456789abcdef0123456789abcdef")
            - Open radar dummy device with sensor_type BGT60TR13C (devicelist -> common_types.py)
            dev = DeviceFmcw(sensor_type = RadarSensor.BGT60TR13C)
        """

        if handle:
            self.handle = handle  # instantiate DeviceFmcw from an existing handle (e.g. dummy)
        else:
            if uuid:
                h = self._cdll.ifx_fmcw_create_by_uuid(uuid.encode("ascii"))
            elif port:
                h = self._cdll.ifx_fmcw_create_by_port(port.encode("ascii"))
            elif sensor_type is not None:  # avoid ambiguity with RadarSensor.BGT60TR13C
                h = self._cdll.ifx_fmcw_create_dummy(sensor_type)
            else:
                h = self._cdll.ifx_fmcw_create()

            # The pointer is returned as an integer. Explicitly casting it to a void pointer
            # ensures it is not truncated by integer handling in certain situations
            self.handle = c_void_p(h)

    def create_dummy_from_device(self) -> 'DeviceFmcw':
        dummy_handle = self._cdll.ifx_fmcw_create_dummy_from_device(self.handle)
        return DeviceFmcw(handle=c_void_p(dummy_handle))

    def get_firmware_information(self) -> dict:
        """Gets information about the firmware of a connected device"""
        info_p = self._cdll.ifx_fmcw_get_firmware_information(self.handle)
        return info_p.contents.to_dict(True)

    def get_sensor_information(self) -> dict:
        """Gets information about the connected device"""
        info_p = self._cdll.ifx_fmcw_get_sensor_information(self.handle)
        info = info_p.contents.to_dict(True)

        for entry in ["lp_cutoff_list", "hp_cutoff_list", "if_gain_list"]:
            info[entry] = create_python_list_from_terminated_list(info[entry])

        return info

    def get_sensor_type(self) -> RadarSensor:
        """Get sensor type of connected device"""
        sensor_type = self._cdll.ifx_fmcw_get_sensor_type(self.handle)
        return sensor_type

    def get_board_uuid(self) -> str:
        """Get the unique id for the radar board"""
        c_uuid = self._cdll.ifx_fmcw_get_board_uuid(self.handle)
        return c_uuid.decode("utf-8")

    def get_temperature(self) -> float:
        """Get the temperature of the device in degrees Celsius

        Note that reading the temperature is not supported for UTR11. An
        exception will be raised in this case.
        """
        temperature = c_float(0)
        self._cdll.ifx_fmcw_get_temperature(self.handle, pointer(temperature))
        return float(temperature.value)

    def save_register_file(self, filename: str) -> None:
        """Save register list to a file"""
        filename_buffer = filename.encode("ascii")
        filename_buffer_p = c_char_p(filename_buffer)
        self._cdll.ifx_fmcw_save_register_file(self.handle, filename_buffer_p)

    def load_register_file(self, filename: str) -> None:
        """Load register list from a file"""
        filename_buffer = filename.encode("ascii")
        filename_buffer_p = c_char_p(filename_buffer)
        self._cdll.ifx_fmcw_load_register_file(self.handle, filename_buffer_p)

    def set_acquisition_sequence(self, first_element: FmcwSequenceElement) -> None:
        """This function tries to configure the radar device to generate the specified
         acquisition sequence"""
        self._cdll.ifx_fmcw_set_acquisition_sequence(self.handle, byref(first_element))

    def get_acquisition_sequence(self) -> FmcwSequenceElement:
        """This function returns the first element of the currently configured
         frame acquisition sequence"""
        first_element = self._cdll.ifx_fmcw_get_acquisition_sequence(self.handle)
        return first_element.contents

    def get_element_duration(self, element: FmcwSequenceElement) -> float:
        """Get duration of a single sequence element
        Returns the wait_time of a delay, the calculated duration of a single chirp,
        or the result of loop duration.
        """
        if element.type == FmcwElementType.IFX_SEQ_LOOP and element.loop.num_repetitions == 0:
            raise RuntimeError("Cannot calculate duration of an infinite loop. Did you mean the loop sub_sequence instead?")
        element_duration = self._cdll.ifx_fmcw_get_element_duration(self.handle, byref(element))
        return float(element_duration)

    def get_sequence_duration(self, first_element: FmcwSequenceElement) -> float:
        """Get duration of sequence
        Returns the sum of the duration of all elements in a sequence.
        """
        if first_element.type == FmcwElementType.IFX_SEQ_LOOP and first_element.loop.num_repetitions == 0:
            raise RuntimeError("Cannot calculate duration of an infinite loop. Did you mean the loop sub_sequence instead?")
        sequence_duration = self._cdll.ifx_fmcw_get_sequence_duration(self.handle, byref(first_element))
        return float(sequence_duration)

    def get_minimum_chirp_repetition_time(self, num_samples: c_uint32, sample_rate_Hz: c_float) -> float:
        """Calculates the minimum possible chirp repetition time, in seconds"""
        chirp_repetition_time = self._cdll.ifx_fmcw_get_minimum_chirp_repetition_time(self.handle, num_samples,
                                                                                      sample_rate_Hz)
        return float(chirp_repetition_time)

    def get_chirp_sampling_bandwidth(self, chirp: FmcwSequenceChirp) -> float:
        """Returns the sampling bandwidth from a chirp"""
        sampling_bandwidth = self._cdll.ifx_fmcw_get_chirp_sampling_bandwidth(self.handle, byref(chirp))
        return float(sampling_bandwidth)

    def get_chirp_sampling_center_frequency(self, chirp: FmcwSequenceChirp) -> float:
        """Returns the center frequency of the sampling bandwidth from a chirp"""
        center_frequency = self._cdll.ifx_fmcw_get_chirp_sampling_center_frequency(self.handle, byref(chirp))
        return float(center_frequency)

    def start_acquisition(self) -> None:
        """Start acquisition of time domain data

        Starts the acquisition of time domain data from the connected device.
        If the data acquisition is already running the function has no effect.
        """
        self._cdll.ifx_fmcw_start_acquisition(self.handle)

    def stop_acquisition(self) -> None:
        """Stop acquisition of time domain data

        Stops the acquisition of time domain data from the connected device.
        If the data acquisition is already stopped the function has no effect.
        """
        self._cdll.ifx_fmcw_stop_acquisition(self.handle)

    def get_next_frame(self, timeout_ms: typing.Optional[int] = None) -> np.ndarray:
        """Retrieve next frame of time domain data from device

        Retrieve the next complete frame of time domain data from the connected
        device.

        The frame is returned as a list of numpy arrays. The size of the list
        corresponds to the size of the acquisition sequence.
        
        For each chirp in the acquisition sequence the frame structure contains a
        cube of real data. The number of dimensions of each cube depends on the
        number of nested loops in the acquisition sequence that surround the chirp
        the cube belongs to.

        Each cube has its data organized in the corresponding dimensions:
        num_virtual_rx_antennas x num_chirps_per_frame x num_samples_per_frame.

        If timeout_ms is given, the exception ErrorTimeout is raised if a
        complete frame is not available within timeout_ms milliseconds.
        """
        frame = self._cdll.ifx_fmcw_allocate_frame(self.handle)
        if timeout_ms:
            self._cdll.ifx_fmcw_get_next_frame_timeout(self.handle, frame, timeout_ms)
        else:
            self._cdll.ifx_fmcw_get_next_frame(self.handle, frame)

        # convert each cube to a multi-dimensional np array
        frame_contents = list()
        for index in range(0, int(frame.contents.num_cubes)):
            cube = frame.contents.cubes[index].contents.to_numpy()
            frame_contents.append(cube)

        self._cdll.ifx_fmcw_destroy_frame(frame)
        return frame_contents

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._close()

    def _close(self):
        """Destroy device handle"""
        if hasattr(self, "handle") and self.handle:
            self._cdll.ifx_fmcw_destroy(self.handle)
            self.handle = None

    def __del__(self):
        try:
            self._close()
        except:
            # just like in C++ exceptions raised in a destructor cannot be catched
            # for this reason we ignore them
            pass
