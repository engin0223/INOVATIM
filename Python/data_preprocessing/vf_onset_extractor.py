"""
Extract VF onset-centered segments from a WFDB RECORDS file.

Usage:
    from data_preprocessing.vf_onset_extractor import extract_vf_segments
    extract_vf_segments(records_file, vf_onset_times, out_file="cardiac_arrest_vf_onset.mat")
"""
import os
import wfdb
import numpy as np
import scipy.io
from scipy.signal import resample

def extract_vf_segments(records_file,
                        vf_onset_times,
                        fs_target=360,
                        pre_sec=30,
                        post_sec=30,
                        out_file="cardiac_arrest_vf_onset.mat",
                        channel_index=0):
    """
    Read RECORDS file (a list of record names). For each record in vf_onset_times,
    extract a VF-centered segment and resample to fs_target, then save to a MAT file.

    Parameters
    ----------
    records_file : str
        Path to a RECORDS file listing record names (one per line).
    vf_onset_times : dict
        Mapping record_name -> onset_seconds (elapsed seconds from start of record).
    fs_target : int
    pre_sec, post_sec : int
        Seconds before/after onset to include.
    out_file : str
        MAT filename to save: contains keys 'X' (list of arrays) and 'records' (list).
    channel_index : int
        Which channel to take from record.p_signal.
    """
    with open(records_file, 'r') as f:
        records = [line.strip() for line in f.readlines()]

    data = []
    record_names = []

    for rec_name in records:
        if rec_name not in vf_onset_times:
            continue

        record_path = os.path.join(os.path.dirname(records_file), rec_name)
        try:
            record = wfdb.rdrecord(record_path)
        except Exception as e:
            print(f"[vf_onset_extractor] Skipping {rec_name}: {e}")
            continue

        if record.p_signal is None or record.p_signal.size == 0:
            print(f"[vf_onset_extractor] Skipping {rec_name}: empty p_signal")
            continue

        ecg = record.p_signal[:, channel_index]
        fs_orig = getattr(record, "fs", None)
        if fs_orig is None:
            print(f"[vf_onset_extractor] Skipping {rec_name}: missing fs")
            continue

        vf_sec = vf_onset_times[rec_name]
        vf_sample = int(vf_sec * fs_orig)

        start = max(0, vf_sample - int(pre_sec * fs_orig))
        end = min(len(ecg), vf_sample + int(post_sec * fs_orig))
        segment = ecg[start:end]

        if len(segment) == 0:
            print(f"[vf_onset_extractor] {rec_name} produced empty segment, skipping")
            continue

        # resample if needed
        if fs_orig != fs_target:
            num_samples = int(len(segment) * fs_target / fs_orig)
            segment = resample(segment, num_samples)

        data.append(segment.astype(np.float32))
        record_names.append(rec_name)
        print(f"[vf_onset_extractor] Record {rec_name}: extracted ~{len(segment)} samples")

    scipy.io.savemat(out_file, {'X': data, 'records': record_names})
    print(f"[vf_onset_extractor] Saved {len(data)} segments -> {out_file}")
    return out_file
