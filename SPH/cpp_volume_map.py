"""Read the exact C++ cached volume map, without resampling its boundary."""
import ctypes
from pathlib import Path
import numpy as np


class CppVolumeMap:
    def __init__(self, cache_path, library_path=None):
        default = Path(__file__).resolve().parent / "output/quant/map-build/Release/cpp_map_bridge.dll"
        self.lib = ctypes.CDLL(str(library_path or default))
        self.lib.map_open.argtypes = [ctypes.c_char_p]
        self.lib.map_open.restype = ctypes.c_void_p
        self.lib.map_error.restype = ctypes.c_char_p
        self.lib.map_close.argtypes = [ctypes.c_void_p]
        array = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
        self.lib.map_query.argtypes = [ctypes.c_void_p, array, ctypes.c_int, array]
        self.lib.map_query.restype = None
        self.handle = self.lib.map_open(str(cache_path).encode("utf-8"))
        if not self.handle:
            raise RuntimeError(self.lib.map_error().decode("utf-8"))

    def query(self, positions):
        positions = np.ascontiguousarray(positions, dtype=np.float64)
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError("positions must have shape (n, 3)")
        result = np.empty((len(positions), 5), np.float64)
        self.lib.map_query(self.handle, positions, len(positions), result)
        return result

    def close(self):
        if self.handle:
            self.lib.map_close(self.handle)
            self.handle = None
