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
from pathlib import Path
import platform


__lib_path = Path(__file__).parent.parent.joinpath("lib")
__system = platform.system()


def get_library_filename(name):
    if __system == "Windows":
        return f"{name}.dll"
    elif __system == "Linux":
        return f"lib{name}.so"
    elif __system == "Darwin":
        return f"lib{name}.dylib"
    else:
        raise RuntimeError(f"System '{__system}' not supported")


def load_library(name: str) -> CDLL:
    """Load SDK library and specify prototypes of common functions

    Load the SDK library and declare prototypes of SDK functions used by all
    wrappers. If system is not supported or the shared library cannot be found,
    an exception is raised.

    Parameters:
        name: Name of SDK library without suffix (e.g., radar_sdk)
    """
    lib_path = __lib_path.joinpath(get_library_filename(name))
    return CDLL(str(lib_path))


def declare_prototype(dll: CDLL, function_name: str, argtypes, restype, errcheck = None) -> None:
    """Declare prototype

    Tell ctypes the argument types and the return type of the C function with
    name function_name.

    Parameters:
        dll: Loaded shared SDK library
        function_name: Name of SDK function
        argtypes: List of arguments or None if no arguments
        restype: Return type or None if function does not return a value
    """
    f = getattr(dll, function_name)
    f.restype = restype
    f.argtypes = argtypes

    # if argument errcheck has been specified, save it as static variable for future use
    if errcheck:
        declare_prototype.errcheck = errcheck
    if hasattr(declare_prototype, "errcheck"):
        f.errcheck = declare_prototype.errcheck

    setattr(dll, function_name, f)
