import numpy as np
from scipy.signal import resample

def pad_segment(segment, desired_len, pad_left_len=0, pad_right_len=0):
    """Pad segment on left/right using edge-value padding."""
    left_pad = np.full(pad_left_len, segment[0]) if pad_left_len > 0 else np.array([])
    right_pad = np.full(pad_right_len, segment[-1]) if pad_right_len > 0 else np.array([])
    return np.concatenate([left_pad, segment, right_pad])

def resample_segment(segment, fs_orig, fs_target):
    """Resample with scipy.signal.resample and return as 1-D array."""
    if fs_orig == fs_target:
        return segment
    num_samples = int(len(segment) * fs_target / fs_orig)
    return resample(segment, num_samples)
