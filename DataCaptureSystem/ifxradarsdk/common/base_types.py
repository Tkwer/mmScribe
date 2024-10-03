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

from ctypes import *

import numpy as np


# maximum number of dimensions supported by ifx_Mda_R_t and ifx_Mda_C_t
IFX_MDA_MAX_DIM = 8


class Complex(Structure):
    """Wrapper for complex number (ifx_Complex_t)"""
    _fields_ = (('data', c_float * 2),
                )


def c_stride(shape: tuple):
    """Convert tuple to C stride array

    The function expects the shape from a numpy array as input and returns
    a C array with the corresponding stride (assuming C order).
    """
    dimensions = len(shape)
    stride = [0] * IFX_MDA_MAX_DIM
    offset = 1
    for k in reversed(range(dimensions)):
        stride[k] = offset
        offset *= shape[k]
    return (c_uint32 * IFX_MDA_MAX_DIM)(*stride)


def c_shape(shape: tuple):
    """Convert tuple to C array (array)

    The function expects the shape from a numpy array as input and returns
    a C array with the shape. The C array has IFX_MDA_MAX_DIM uint32 values.
    If the input shape is smaller than IFX_MDA_MAX_DIM the C array will
    be padded with zeros.
    """
    dimensions = len(shape)
    py_shape = list(shape) + [0] * (IFX_MDA_MAX_DIM - dimensions)
    return (c_uint32 * IFX_MDA_MAX_DIM)(*py_shape)


def truncate_list_at_zero(l: tuple):
    """Truncate list at first 0 value found

    [1,2,3] => [1,2,3]
    [1,2,3,0,0] => [1,2,3]
    [1,2,3,0,4] => [1,2,3]
    """
    new_list = []
    for v in l:
        if v == 0:
            break
        new_list.append(v)
    return new_list


class MdaReal(Structure):
    """Wrapper for the ifx_Mda_R_t structure"""
    _fields_ = (('dimensions', c_uint32),
                ('data', POINTER(c_float)),
                ('shape', c_uint32 * IFX_MDA_MAX_DIM),
                ('stride', c_uint32 * IFX_MDA_MAX_DIM),
                ('flags', c_uint32),
                )

    @classmethod
    def from_numpy(cls, np_arr: np.ndarray):
        """Create numpy array from ifx_Mda_R_t object"""
        shape = np_arr.shape
        dimensions = len(shape)
        if dimensions > IFX_MDA_MAX_DIM:
            raise ValueError("too many dimensions")

        # copy array, make sure the array is in C order and uses floats
        np_arr = np.array(np_arr, dtype=np.float32, order="C")

        # We do not copy data but create a view. It is crucial that the memory
        # of the arrary np_arr is not released as long as this object
        # exists. For this reason we assign np_arr to this object. This way
        # the memory of np_arr is not released before our current object is
        # released.
        data = np_arr.ctypes.data_as(POINTER(c_float))

        flags = 0
        arr = MdaReal(dimensions, data, c_shape(shape), c_stride(shape), flags)
        arr.np_arr = np_arr  # avoid that memory of np_arr is freed
        return arr

    def to_numpy(self) -> np.ndarray:
        """Convert ifx_Mda_R_t type to a numpy array"""
        shape = truncate_list_at_zero(self.shape)
        data = np.ctypeslib.as_array(self.data, shape)
        return np.array(data, order="C", copy=True)


class MdaComplex(Structure):
    """Wrapper for the ifx_Mda_C_t structure"""
    _fields_ = (('dimensions', c_uint32),
                ('data', POINTER(Complex)),
                ('shape', c_uint32 * IFX_MDA_MAX_DIM),
                ('stride', c_uint32 * IFX_MDA_MAX_DIM),
                ('flags', c_uint32),
                )

    @classmethod
    def from_numpy(cls, np_arr: np.ndarray):
        """Create numpy array from ifx_Mda_C_t object"""
        shape = np_arr.shape
        dimensions = len(shape)
        if dimensions > IFX_MDA_MAX_DIM:
            raise ValueError("too many dimensions")

        # copy array, make sure the array is in C order and uses floats
        np_arr = np.array(np_arr, dtype=np.complex64, order="C")

        # We do not copy data but create a view. It is crucial that the memory
        # of the arrary np_arr is not released as long as this object
        # exists. For this reason we assign np_arr to this object. This way
        # the memory of np_arr is not released before our current object is
        # released.
        data = np_arr.ctypes.data_as(POINTER(Complex))

        flags = 0
        arr = MdaComplex(dimensions, data, c_shape(shape), c_stride(shape), flags)
        arr.np_arr = np_arr  # avoid that memory of np_arr is freed
        return arr

    def to_numpy(self) -> np.ndarray:
        """Convert ifx_Mda_C_t type to a numpy array"""
        shape = truncate_list_at_zero(self.shape)
        data = np.ctypeslib.as_array(cast(self.data, POINTER(c_float)), (2 * np.prod(shape),))
        arr_1d = np.array(data, order="C", copy=False)
        return np.array(arr_1d[::2] + 1j * arr_1d[1::2], order="C", dtype=np.complex64, copy="True").reshape(shape)


class ifxStructure(Structure):
    """Wrapper for C structs

    This is a wrapper around ctypes.Structure with more convenience like
    converting a structure to a dictionary.
    """

    def _initialize_fields(self, dictionary: dict) -> None:
        for (name, _) in self._fields_:
            setattr(self, name, dictionary[name])

    def _get_field(self, field_name: str, decode_byte_str: bool = False) -> object:
        """Get value of field

        Return string representation of member. This function might not work
        for more complex types like pointers or nested structures.

        Parameters:
            field_name: Name of field
            decode_byte_str: If true decode byte arrays into string assuming ASCII encoding
        """
        v = getattr(self, field_name)
        if decode_byte_str and type(v) == bytes:
            return v.decode("ascii")
        else:
            return v

    def _get_field_as_str(self, field_name: str) -> str:
        """Convert structure to dictionary

        Return string representation of field. This function might not work for
        more complex types like pointers or nested structures.

        Parameters:
            field_name: Name of field
        """
        return str(getattr(self, field_name))

    def to_dict(self, decode_byte_str: bool = False) -> dict:
        """Convert structure to dictionary

        For each member _get_field_as_str is called.

        Parameters:
            decode_byte_str: If true decode byte arrays into string assuming ASCII encoding
        """
        return {field_name: self._get_field(field_name, decode_byte_str) for field_name, _ in self._fields_}

    def __repr__(self) -> str:
        """Return string representation of object"""
        l = [f"{self.__class__.__name__}:"]
        for field_name, field_type in self._fields_:
            l.append(f"    {field_name}: {self._get_field_as_str(field_name)}")
        return "\n".join(l)
