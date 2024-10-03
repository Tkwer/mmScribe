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
import typing

from .base_types import MdaComplex, MdaReal
from .cdll_helper import declare_prototype, load_library
from .exceptions import get_exception


def check_error(result, func, arguments):
    """Raise exception on error or return the result

    The function checks if an error
    occurred inside the SDK by calling ifx_error_get_and_clear(). If an error
    occurred the corresponding Python exception is raised. If no error occurred,
    the function forwards the result.

    Parameters:
        result: Return value from CDLL call
        fund: Original function object reference
        arguments: list of provided function arguments
    """

    error_code = _cdll.ifx_error_get_and_clear()
    if error_code:
        error_description = _cdll.ifx_error_to_string(error_code).decode("ascii")
        raise get_exception(error_code, error_description)

    return result


def __load_cdll(errcheck : typing.Callable) -> CDLL:
    """Load SDK library and specify prototypes of common functions

    Load the SDK library and declare prototypes of SDK functions used by all
    wrappers. If system is not supported or the shared library cannot be found,
    an exception is raised.

    Parameters:
        libname: Name of SDK library without suffix (e.g., radar_sdk)
    """
    dll = load_library("sdk_base")

    # declare error related functions first (this has to happen before any other functions in related DLLs are declared)
    # errcheck=None since for error checking functions themselves we do not need error checking
    declare_prototype(dll, "ifx_error_get_and_clear", None, c_int, None)
    declare_prototype(dll, "ifx_error_to_string", [c_int], c_char_p, None)

    # version
    declare_prototype(dll, "ifx_sdk_get_version_string", None, c_char_p, errcheck)  # specify check_error as errcheck to be used for current and any future declaration
    declare_prototype(dll, "ifx_sdk_get_version_string_full", None, c_char_p)

    # destroy multi-dimensional arrays
    declare_prototype(dll, "ifx_mda_destroy_r", [POINTER(MdaReal)], None)
    declare_prototype(dll, "ifx_mda_destroy_c", [POINTER(MdaComplex)], None)

    # list
    declare_prototype(dll, "ifx_list_destroy", [c_void_p], None)
    declare_prototype(dll, "ifx_list_size", [c_void_p], c_size_t)
    declare_prototype(dll, "ifx_list_get", [c_void_p, c_size_t], c_void_p)

    # memory management
    declare_prototype(dll, "ifx_mem_free", [c_void_p], None)

    declare_prototype(dll, "ifx_uuid_to_string", [POINTER(c_uint8), c_char_p], None)

    return dll


_cdll = __load_cdll(check_error)


def get_version(full : bool = False) -> str:
    """Return SDK version string

    If full is False, the major, minor, and patch number of the SDK version 
    is returned as a string in the format "X.Y.Z". If full is true, the full
    version information including git tag and git commit is returned.
    """
    if full:
        return _cdll.ifx_sdk_get_version_string_full().decode("ascii")
    else:
        return _cdll.ifx_sdk_get_version_string().decode("ascii")


def get_version_full() -> str:
    """Return full SDK version string including git tag from which it was build"""
    return _cdll.ifx_sdk_get_version_string_full().decode("ascii")


def move_ifx_list_to_python_list(ifx_list: c_void_p, func : typing.Callable) -> list:
    result = []
    size = _cdll.ifx_list_size(ifx_list)
    for i in range(size):
        p = _cdll.ifx_list_get(ifx_list, i)
        result.append(func(p))
    _cdll.ifx_list_destroy(ifx_list)

    return result


def ifx_mem_free(ptr):
    _cdll.ifx_mem_free(ptr)


def ifx_mda_destroy_r(mda):
    _cdll.ifx_mda_destroy_r(mda)


def ifx_mda_destroy_c(mda):
    _cdll.ifx_mda_destroy_c(mda)
