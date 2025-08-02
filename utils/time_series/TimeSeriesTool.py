import numpy as np


class TimeSeriesTool:

    @classmethod
    def match_range_indexes(cls, value, array, start_idx, end_idx):
        return (np.flatnonzero(array[start_idx: end_idx + 1] == value) + start_idx).tolist()

    @classmethod
    def match_first_index(cls, value, array):
        if np.any(array == value):
            return np.argmax(array == value)
        return -1

    @classmethod
    def match_last_index(cls, value, array):
        if array.size == 0:
            return -1
        reversed_array = array[::-1]
        if np.any(reversed_array == value):
            return len(array) - 1 - np.argmax(reversed_array == 1)
        return -1
