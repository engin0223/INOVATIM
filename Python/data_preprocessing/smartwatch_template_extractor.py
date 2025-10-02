"""
Extract short heartbeat-centered windows from a collection of RECORDS (walks folders),
compute cosine-distance scores against templates stored in a MATLAB v7.3 file, and save to MAT.

Usage:
    from data_preprocessing.smartwatch_template_extractor import extract_smartwatch_segments
    extract_smartwatch_segments(main_folder, mlii_matfile, out_file="ecg_smartwatch_all_template_scores.mat")
"""
import os
import numpy as np
import wfdb
import scipy.io
import h5py
from scipy.signal import resample
from wfdb.processing import xqrs_detect
from scipy.spatial.distance import cosine
from .utils import resample_segment

def load_templates_from_mat(mlii_matfile):
    templates = {}
    with h5py.File(mlii_matfile, 'r') as f:
        X_templates = np.array(f['X_templates'])
        Y_templates = np.array(f['Y_templates'])
        # Y_templates are MATLAB cellstr references: convert each to string
        for i in range(Y_templates.shape[0]):
            ref = Y_templates[i, 0]
            ref = ref.decode() if isinstance(ref, bytes) else ref
            # h5py stores char codes under the ref dataset
            sym_codes = np.array(f[ref][:]).flatten()
            sym = ''.join([chr(int(c)) for c in sym_codes])
            temp = np.array(X_templates[:, i]).flatten()
            templates.setdefault(sym, []).append(temp)
    for k in list(templates.keys()):
        templates[k] = np.vstack(templates[k])
    return templates

def assign_symbol_scores(segment, templates):
    seg_norm = (segment - np.mean(segment)) / (np.std(segment) + 1e-6)
    scores = {}
    for sym, template_set in templates.items():
        best_score = float('inf')
        for temp in template_set:
            if len(temp) != len(seg_norm):
                continue
            s = cosine(seg_norm, temp)
            if s < best_score:
                best_score = s
        scores[sym] = best_score
    return scores

def find_record_files(folder):
    record_files = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.lower() == 'records':
                record_files.append(os.path.join(root, file))
    return record_files

def extract_smartwatch_segments(main_folder,
                                mlii_matfile,
                                fs_target=360,
                                pre_sec=0.25,
                                post_sec=0.25,
                                out_file='ecg_smartwatch_all_template_scores.mat',
                                enable_plot=False):
    templates = load_templates_from_mat(mlii_matfile)
    print(f"[smartwatch_extractor] Loaded {len(templates)} template symbols")

    data = []
    record_names = []
    annotations_list = []

    record_files = find_record_files(main_folder)

    for rec_file in record_files:
        folder_path = os.path.dirname(rec_file)
        with open(rec_file, 'r') as f:
            records = [line.strip() for line in f.readlines()]

        for rec_name in records:
            record_path = os.path.join(folder_path, rec_name)
            try:
                record = wfdb.rdrecord(record_path)
            except Exception as e:
                print(f"[smartwatch_extractor] Skipping {rec_name}: {e}")
                continue

            ecg = record.p_signal[:, 0] if record.p_signal is not None else None
            fs_orig = getattr(record, "fs", None)
            if ecg is None or fs_orig is None:
                print(f"[smartwatch_extractor] {rec_name} missing data/fs, skipping")
                continue
            if len(ecg) < 2 or np.std(ecg) < 1e-6:
                print(f"[smartwatch_extractor] {rec_name} signal too short/flat, skipping")
                continue

            try:
                rpeaks = xqrs_detect(sig=ecg, fs=fs_orig)
            except Exception as e:
                print(f"[smartwatch_extractor] R-peak detection failed for {rec_name}: {e}")
                continue

            if len(rpeaks) == 0:
                continue

            desired_len_orig = int((pre_sec + post_sec) * fs_orig)
            desired_len_target = int((pre_sec + post_sec) * fs_target)

            for hb_sample in rpeaks:
                start = hb_sample - int(pre_sec * fs_orig)
                end = hb_sample + int(post_sec * fs_orig)
                start_clip = max(0, start)
                end_clip = min(len(ecg), end)
                segment = ecg[start_clip:end_clip].astype(np.float32)

                # padding if necessary
                if len(segment) < desired_len_orig:
                    pad_left = start_clip - start
                    pad_right = end - end_clip
                    left_pad = np.full(pad_left, segment[0]) if pad_left > 0 else np.array([])
                    right_pad = np.full(pad_right, segment[-1]) if pad_right > 0 else np.array([])
                    segment = np.concatenate([left_pad, segment, right_pad])
                segment = segment[:desired_len_orig]

                # resample
                if fs_orig != fs_target:
                    segment = resample(segment, desired_len_target)
                else:
                    # ensure exact length
                    if len(segment) != desired_len_target:
                        segment = np.resize(segment, desired_len_target)

                symbol_scores = assign_symbol_scores(segment, templates)

                data.append(segment)
                record_names.append(rec_name)
                annotations_list.append(symbol_scores)

    # convert to savable arrays
    X = np.vstack(data) if len(data) > 0 else np.empty((0, int((pre_sec+post_sec)*fs_target)))
    Y = np.array(annotations_list, dtype=object)
    scipy.io.savemat(out_file, {'X': X, 'records': np.array(record_names, dtype=object), 'annotations': Y})
    print(f"[smartwatch_extractor] Saved {len(data)} segments -> {out_file}")
    return out_file
